//! A bundle's pipelines compiled side by side.
//!
//! A pipeline's compile is the backend's shader compiler at work: seconds
//! for a long kernel (a terrain program inlined into a walk), and a bundle
//! holds dozens. One after another they kept a new world from its first
//! chunk for the sum of them; side by side, for about the longest one.
//! [`Kernels`] collects a bundle's pipelines as they are named
//! ([`Kernels::add`]) and compiles them together ([`Kernels::compile`]) on
//! the thread pool, inside the bundle's own compile job
//! (`helio_core::compile::Compiling`): the pool is the host's, and a job
//! that waited on further jobs would wait on the host's few long-task
//! workers it is itself holding.

use std::sync::{Arc, Mutex, PoisonError};

/// One pipeline's build: what it could not build, if so.
type Job<'a> = Box<dyn FnOnce() -> Result<(), String> + Send + 'a>;

/// The pipelines of one bundle, compiled together.
pub(super) struct Kernels<'a> {
    device: &'a wgpu::Device,
    jobs: Vec<Job<'a>>,
}

/// A pipeline of a [`Kernels`]: taken once they are compiled.
pub(super) struct Kernel<T>(Arc<Mutex<Option<T>>>);

impl<T> Kernel<T> {
    /// The pipeline, after [`Kernels::compile`].
    pub fn take(self) -> T {
        self.0
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .take()
            .expect("taken once, after its bundle compiled")
    }
}

impl<'a> Kernels<'a> {
    pub fn new(device: &'a wgpu::Device) -> Self {
        Self {
            device,
            jobs: Vec::new(),
        }
    }

    /// A pipeline `build` makes, compiled with the others.
    pub fn add<T: Send + 'a>(&mut self, build: impl FnOnce() -> T + Send + 'a) -> Kernel<T> {
        let slot = Arc::new(Mutex::new(None));
        let (out, device) = (slot.clone(), self.device);
        self.jobs.push(Box::new(move || {
            // Error scopes are a thread's: each build answers for itself,
            // on whichever thread runs it.
            let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
            let built = build();
            let error = now(scope.pop());
            *out.lock().unwrap_or_else(PoisonError::into_inner) = Some(built);
            error.map_or(Ok(()), |error| Err(error.to_string()))
        }));
        Kernel(slot)
    }

    /// Compile every pipeline added, side by side: as many at a time as a
    /// quarter of the pool's threads (2 to 8, so the pool keeps serving the
    /// frame), each taking the next one not started. Panics with the first
    /// failure's reason: the bundle's compile job reports it as the bundle's
    /// (`Compiling::failed`).
    pub fn compile(self) {
        use rayon::prelude::*;
        let lanes = (rayon::current_num_threads() / 4)
            .clamp(2, 8)
            .min(self.jobs.len().max(1));
        let next = Mutex::new(self.jobs.into_iter());
        let failures: Vec<String> = (0..lanes)
            .into_par_iter()
            .flat_map_iter(|_| {
                let mut failed = Vec::new();
                loop {
                    let job = next.lock().unwrap_or_else(PoisonError::into_inner).next();
                    match job {
                        Some(job) => failed.extend(job().err()),
                        None => break failed,
                    }
                }
            })
            .collect();
        if let Some(reason) = failures.first() {
            panic!("{reason}");
        }
    }
}

/// A future that is ready when asked (a native device's error scope).
fn now<T>(future: impl std::future::Future<Output = Option<T>>) -> Option<T> {
    let mut future = std::pin::pin!(future);
    let mut context = std::task::Context::from_waker(std::task::Waker::noop());
    match future.as_mut().poll(&mut context) {
        std::task::Poll::Ready(value) => value,
        std::task::Poll::Pending => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn device() -> Option<wgpu::Device> {
        let instance = wgpu::Instance::new(helio_core::wgpu_instance_descriptor());
        let adapter = pollster::block_on(instance.request_adapter(&Default::default())).ok()?;
        pollster::block_on(adapter.request_device(&Default::default()))
            .ok()
            .map(|(device, _)| device)
    }

    /// Every pipeline added is built once and taken as its own, several at
    /// a time: a bundle takes about its longest build, not their sum.
    #[test]
    fn a_bundles_builds_run_side_by_side_and_keep_their_places() {
        let Some(device) = device() else { return };
        let (running, most) = (AtomicUsize::new(0), AtomicUsize::new(0));
        let mut kernels = Kernels::new(&device);
        let slots: Vec<Kernel<usize>> = (0..24)
            .map(|n| {
                let (running, most) = (&running, &most);
                kernels.add(move || {
                    let now = running.fetch_add(1, Ordering::SeqCst) + 1;
                    most.fetch_max(now, Ordering::SeqCst);
                    std::thread::sleep(std::time::Duration::from_millis(20));
                    running.fetch_sub(1, Ordering::SeqCst);
                    n * 7
                })
            })
            .collect();
        let started = std::time::Instant::now();
        kernels.compile();
        let took = started.elapsed();
        for (n, slot) in slots.into_iter().enumerate() {
            assert_eq!(slot.take(), n * 7);
        }
        let lanes = (rayon::current_num_threads() / 4).clamp(2, 8);
        assert!(
            most.load(Ordering::SeqCst) <= lanes,
            "no more than its lanes at a time"
        );
        if rayon::current_num_threads() >= 2 {
            assert!(most.load(Ordering::SeqCst) >= 2, "side by side");
            assert!(
                took.as_millis() < 24 * 20,
                "{took:?} for 24 builds of 20 ms"
            );
        }
    }

    /// A pipeline the backend rejects fails the bundle with the reason,
    /// whichever thread built it.
    #[test]
    fn a_rejected_pipeline_is_the_bundles_failure() {
        let Some(device) = device() else { return };
        let failure = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let mut kernels = Kernels::new(&device);
            let _fine = kernels.add(|| 1);
            let _bad = kernels.add(|| {
                device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("not wgsl"),
                    source: wgpu::ShaderSource::Wgsl("fn broken( {".into()),
                })
            });
            kernels.compile();
        }));
        let reason = failure.expect_err("the bundle fails");
        let reason = reason.downcast_ref::<String>().cloned().unwrap_or_default();
        assert!(!reason.is_empty(), "with the backend's reason");
    }
}
