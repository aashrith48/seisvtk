# F3 Demo 2023 — cleaned subset

F3 block, offshore Netherlands. Extracted from the OpendTect F3 Demo project;
proprietary CBVS volumes and OpendTect workflow files removed. **CC BY-SA 3.0.**

## Seismic — `seismic/Seismic_data.sgy`

Full-stack migrated, **two-way time**.

| | |
|---|---|
| Inlines | 651 (100–750, step 1), header byte **189** |
| Crosslines | 951 (300–1250, step 1), header byte **193** |
| Samples | 462 (4–1848 ms, **4 ms**) |
| Traces | 600,515 |
| Format | **2-byte signed int**, nominal range ±8000 |
| CRS | EPSG:23031, ED50 / UTM 31N, ~25 × 25 m bins |

**Gotchas**

1. `segyio.open()` returns *unstructured* — 18,586 traces (3%) are missing, all in
   **inlines 701–750** at the survey edge. Inlines 100–700 are complete at 951 traces.
   Either crop to IL 100–700 (dense 601 × 951 × 462, 1.06 GB float32) or open with
   `ignore_geometry=True` and scatter traces into a padded array.
2. Samples are `int16`. Convert before any attribute math or you will overflow.

**Grid ↔ world transform** (verified against trace headers to 0.1 m):

```
X = 598408.2476 - 0.6980137931·IL + 24.99024752·XL
Y = 6070847.887 + 24.98965517·IL + 0.6984323432·XL
```

## Horizons — `horizons/`

7 horizons, plain ASCII `X Y TWT_ms`, no inline/crossline.

| Horizon | TWT range (ms) | Points |
|---|---|---|
| Shallow | 332.8 – 499.0 | 606,690 |
| FS8 | 415.4 – 860.5 | 591,260 |
| FS7 | 457.2 – 1026.2 | 607,632 |
| Truncation | 458.2 – 1070.8 | 592,150 |
| FS6 | 482.5 – 1007.4 | 303,622 |
| Top Foresets | 635.9 – 1088.9 | 456,943 |
| MFS4 | 528.3 – 1097.9 | 592,177 |

`FS4` is **not** included — it existed only in OpendTect `.hor` format.

## Faults — `faults/FaultA.txt`

Single interpreted fault. Usable as geometry overlay or fault-detection label.

## Wells — `wells/`

4 vertical wells: **F02-1, F03-2, F03-4, F06-1**.

| Dir | Contents |
|---|---|
| `logs/` | Processed LAS: `DEPTH RHOB DT GR AI AI_rel PHIE` (identical curve set in all 4) |
| `raw_F02-01/` | **Unprocessed** F02-1 logs: `DEPTH CALI RHOB GR DT DT UNKNOWN`. Has CALI, lacks AI/PHIE — genuinely different from `logs/`, not a duplicate |
| `tracks/` | 2-point well tracks (surface X/Y → TD) |
| `markers/` | Stratigraphic tops per well + `F3-well-markers.xls` |
| `checkshots/` | Measured time–depth: `MD_m  TWT_SECONDS` |
| `dt_models/` | Sonic-integrated time–depth: `MD_m  TWT_MILLISECONDS` |
| `dt_models_tvdss/` | Same, referenced to TVDSS |
| `tie_setup/` | OpendTect well-tie *settings* (which log/wavelet was used) — parameters, not curves |

**Time–depth gotchas**

1. **Units differ between the two sources.** `checkshots/` is in *seconds*
   (0.473–3.234); `dt_models/` is in *milliseconds* (503.7–3258.9). Mixing them is a
   1000× error.
2. **Checkshots are not monotonic.** F03-2 and F03-4 each contain a row where MD steps
   backwards (F03-4 jumps 2980 → 2277 m). Sort by MD before interpolating.
3. **Duplicate MD values** — F03-4 has 547.75 m three times. Deduplicate or a naive
   slope calculation divides by zero.
4. The two sources disagree by ~20 ms (~2%) at 1 km — ordinary sonic drift. Anchor on
   the checkshot; use the DT model only to interpolate between checkshot points.

**Seismic ↔ log overlap.** Logs do not span the volume:

| Well | Log TWT coverage | Overlap with 4–1848 ms |
|---|---|---|
| F02-1 | 544 – 3234 ms | 544 – 1848 (clip base) |
| F03-2 | 485 – 1732 ms | full |
| F03-4 | 473 – 2907 ms | 473 – 1848 (clip base) |
| F06-1 | 593 – 1221 ms | full |

The top ~470 ms of the volume has **no log coverage**, so the `Shallow` horizon
(332.8 ms) cannot be well-tied.

## Metadata — `meta/`

- `survey.txt` — original OpendTect `.survey`: the affine above, CRS, ranges
- `Velocity_functions.txt` — 46,669 rows, `CDP-X CDP-Y Time(ms) Vrms Vint Vavg Depth(m)`.
  Only accessible depth-conversion path. Its own header states *"example velocities, not
  measured velocities"* — treat any depth conversion built from it as illustrative
- `Chimneys_yes.pck` / `Chimneys_no.pck` — hand-labelled positive/negative point sets

## Note on combining with Volve

F3 is **time** (0–1848 ms); Volve is **depth** (0–4500 m). Not directly stackable, and
the velocity field here is not rigorous enough to convert honestly. Prefer a per-dataset
domain flag over resampling through example velocities. Licences also differ — F3 is
CC BY-SA 3.0, Volve CC BY-NC-SA 4.0 (**NC**).
