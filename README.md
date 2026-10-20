# FallDar: Selective CSI Refinement Feasibility Study

A preliminary experiment for *Selective CSI Refinement for Bandwidth-Efficient Wi-Fi Fall Detection*. It checks whether additional CSI can correct fall misses caused by strong compression.

## Why FallDar

| Requirement | FallDar | CSI-Bench (Kaggle fall subset) |
| --- | --- | --- |
| Complex CSI (phase), needed to simulate what an edge device transmits | Yes, 30 subcarriers x 3 antennas | No, amplitude only |
| Continuous recordings / timestamps, needed for online refinement | Yes, raw `.dat` files | No, fixed segments |
| Fall labels | 606 fall / 1,049 non-fall, 5 fall types | Binary only |
| Recording IDs for leakage-free splits | Yes, from file names | Not available |

FallDar is the primary dataset. Its main limitation is a single living room, so CSI-Bench is kept as a secondary benchmark for generalisation.

## What the code does to the data

1. **Load**: reads the 1,655 labelled windows in `denoisemat/` directly from `FallDar.zip`. Each window is complex CSI of shape (2000 packets, 30 subcarriers, 3 antennas).
2. **Downsample**: averages every 4 packets, giving 500 time steps.
3. **Compress**: applies a 2-D DCT (time x subcarrier) to the real and imaginary parts and keeps only a low-frequency block. Eight rates are tested, from 1x to 114x. The blocks are nested, so refinement only sends the missing coefficients.
4. **Features**: takes the amplitude of the reconstructed CSI, removes each stream's static mean, and normalises each window.
5. **Split**: groups windows by recording, so windows cut from the same recording never appear in both training and test sets. Results are averaged over 3 random splits.
6. **Evaluate** with a small 1-D CNN:
   - **Fixed-rate**: one model per compression rate.
   - **Oracle refinement**: misclassified windows are refined to a higher rate (upper bound).
   - **Same-rate control**: a second model at the same rate, to separate the information gain from the ensemble effect.

## Key result

At 114x compression, fall recall drops from 0.956 to 0.860. Refining only the misclassified 11% of windows to full CSI restores it to 0.992, at about 12% of the full transmission. The same-rate control reaches 0.931, so +0.061 recall comes from the extra CSI itself.

