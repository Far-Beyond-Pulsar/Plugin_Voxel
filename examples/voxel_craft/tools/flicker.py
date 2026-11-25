import sys, glob, numpy as np
from scipy import ndimage
from PIL import Image
d = sys.argv[1]
poses = sorted({p.split('shot_')[1][:2] for p in glob.glob(d + '/shot_*_*.png')})
for pose in poses:
    files = sorted(glob.glob(f'{d}/shot_{pose}_*.png'))
    fr = [np.asarray(Image.open(f).convert('RGB'), dtype=np.float32) / 255.0 for f in files]
    lum = [f @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32) for f in fr]
    diffs = [np.abs(lum[i + 1] - lum[i]) for i in range(len(lum) - 1)]
    mean = np.mean([x.mean() for x in diffs])
    big = np.mean([(x > 0.02).mean() for x in diffs])
    # Per-pixel temporal std over the burst.
    std = np.std(np.stack(lum), axis=0)
    # Frames where much of the image changes at once (over 5% of the pixels
    # by more than 2%): not flicker but an event, such as the NVIDIA lost
    # G-buffer writes (8x8-pixel blocks showing the background, bug
    # 6939141), which a burst should be retaken over.
    events = [i + 1 for i, x in enumerate(diffs) if (x > 0.02).mean() > 0.05]
    # Speckles: pixels brighter by over 40/255 than the frames before and
    # after, in clusters of 4 or more (a leaf cell or two lit for a frame;
    # lone pixels are edges aliasing under motion), a frame. The canopy gate:
    # pose 12 (the crowns just after a look around) at 1080p at most 20
    # (the base before the history fix: 55-69; with it: 12-14).
    l8 = np.stack(lum) * 255.0
    pops = (l8[1:-1] - np.maximum(l8[:-2], l8[2:])) > 40
    speck = 0
    for f in pops:
        lab, k = ndimage.label(f, np.ones((3, 3)))
        if k:
            sizes = ndimage.sum(f, lab, range(1, k + 1))
            speck += sizes[sizes >= 4].sum()
    speck /= max(len(pops), 1)
    print(f"pose {pose}: mean |dL| {mean:.5f}, pixels changing >2% {big*100:.2f}%, temporal std mean {std.mean():.5f}, p99 {np.percentile(std, 99):.4f}, speckle px/frame {speck:.1f}, event frames {events}")
    np.save(f'{d}/std_{pose}.npy', std)
