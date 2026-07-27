# Teapot Dome 3D (NPR-3) — cleaned subset

Naval Petroleum Reserve No. 3, Teapot Dome, Natrona County, Wyoming. Extracted from
RMOTC's `rmotc.tar`; ESRI geodatabase, core imagery and FMI logs removed.

**Terms:** non-proprietary. Acknowledge **RMOTC and the U.S. Department of Energy** as the
data source in any presentation or publication. See `meta/Seismic_read_me.txt` and
`meta/conditions for use.docx`.

## Seismic — `seismic/filt_mig.sgy`

The textual header states it plainly: *"THIS IS THE FILTERED POST STACK MIGRATION"* —
i.e. the final post-stack migrated image, band-pass filtered after migration.

| | |
|---|---|
| Domain | **TWT**, 0–3000 ms, **2 ms**, 1501 samples |
| Inlines | 345 (1–345) |
| Crosslines | 188 (1–188) |
| Traces | 64,860 = 345 × 188 — **perfectly dense, no gaps** |
| Bins | 110 × 110 ft (verified from coordinates) |
| Format | 4-byte **IBM** float |
| CRS | Wyoming East Central State Plane, **NAD 1927**, feet |
| Dense cube | **0.39 GB** float32 |

**Gotcha — the header bytes are swapped relative to the SEG-Y standard:**

```
bytes 181/185  (nominally CDP_X / CDP_Y)     actually hold  INLINE / XLINE
bytes 189/193  (nominally INLINE / XLINE)    actually hold  X / Y coordinates
```

```python
with segyio.open("seismic/filt_mig.sgy", "r", iline=181, xline=185) as s:
    cube = segyio.tools.cube(s)          # (345, 188, 1501)
```

Corner coordinates verified against the textual header:

| Inline | Xline | X | Y |
|---|---|---|---|
| 1 | 1 | 788937 | 938846 |
| 1 | 188 | 809502 | 939334 |
| 345 | 1 | 788039 | 976675 |

Also **ignore the coordinate scalar of −10** in the trace headers — the X/Y values are
already unscaled State Plane feet and match the textual header verbatim.

Reference datum: 5500 ft, replacement velocity 9500 ft/s.

## 2D lines — `seismic/2d_lines/`

Five migrated 2D lines (A–E) plus navigation. Small; kept for completeness.

## Horizons — `horizons/3DHorizons.xyz`

Tab-separated, 241,587 picks. **Best-structured horizons of any survey here** — carries
grid coordinates *and* world coordinates *and* time, so no transform is needed:

```
inline  xline  X  Y  Horizon  Time  Velocity  Depth  Amplitude
```

| Horizon | Picks | TWT max (s) |
|---|---|---|
| Carlile | 31,112 | 0.710 |
| KF2 | 27,620 | 0.815 |
| FallRiver | 25,462 | 0.999 |
| Lakota/Morrison | 26,960 | 0.999 |
| CrowMountain | 27,811 | 1.095 |
| RedPeak | 27,176 | 1.119 |
| Tensleep | 25,188 | 1.218 |
| TensleepBbase | 25,006 | 1.227 |
| Basement | 25,252 | 1.389 |

**Two traps:**

1. `Velocity` and `Depth` are `-999.99` in **all 241,587 rows** — placeholder columns
   carrying no data. The only depth-conversion inputs are `meta/TimeDepthTables.xls` and
   the separately-downloadable `npr3_dmo.vel`.
2. Every horizon's minimum `Time` is exactly `0.000` — that is a **null-pick encoding**,
   not a pick at the surface. Filter `Time > 0` before gridding.

`Time` is in **seconds**; the seismic axis is in **ms**.

## Wells — `wells/`

| Dir | Contents |
|---|---|
| `logs/` | **1,200 LAS files** in `Shallow_LAS_files/` and `Deeper_LAS_files/`, one subdirectory per API number |
| `headers/` | `TeapotDomeWellHeaders02-09-10.xlsx` — **1,317 wells**: API, name, Northing/Easting, total depth, KB datum elevation, ground elevation, status, spud/completion dates |
| `surveys/` | `DirectionalSurveys_020910.xlsx` — **42,441 rows** of API / MD(ft) / inclination / azimuth |
| `tops/` | `TeapotDomeFormationLogTops.xls` (**7,285 tops**: API / well / formation alias / Top MD), plus `TeapotGeologic_Column.xls` and `FormationCodes.doc` |
| `production/` | `NPR-3_ProductionData_thru11-30-05.xls` |

**Both spreadsheets have their real header on row 2** — read with `header=1`, or you get
`Unnamed: N` columns.

**These wells are deviated.** Unlike F3 and Stratton (vertical), Teapot ships directional
surveys, so well paths must be computed from MD/inclination/azimuth (minimum curvature)
before anything can be placed in the volume. You cannot assume MD ≈ TVD.

**Well and seismic coordinates share a CRS.** Header Easting (794,811–808,959) and
Northing (950,092–974,652) fall inside the seismic X (788,039–809,502) and
Y (938,846–977,163) — both Wyoming East Central State Plane NAD27 feet, with
seismic X = Easting and seismic Y = Northing. No transform needed.

**Ties are the weak link.** Formation tops are in **MD feet only — no two-way time**
(unlike Stratton's TABLE2, which gave TWT directly). To tie wells to seismic you must:

```
MD  --(directional survey)-->  TVD  --(KB datum elev)-->  TVDSS
    --(meta/TimeDepthTables.xls)-->  TWT
```

`meta/TimeDepthTables.xls` is described as *"select time/depth tables"*, so expect
coverage for only a handful of the 1,317 wells. Reading it needs `pip install xlrd`
(legacy `.xls`).

## Core — `core/`

Only the machine-readable parts were kept: `logs/` (10 LAS, wireline and FMI-derived) and
`analyses/` (core porosity/permeability and FMI dip spreadsheets). The 323 MB of core
photographs, TIFFs, FMI `.pds` files, a 96 MB PowerPoint album and a Windows `.exe` viewer
were deleted.

## Metadata — `meta/`

`TimeDepthTables.xls` (time–depth) · `NPR3_FieldBoundary.txt` (ASCII field outline) ·
`Seismic_read_me.txt` · `WellDataset_read_me.txt` · `GIS_read_me.txt` ·
`conditions for use.docx` · `teapot_processing.doc` (3D processing parameters) ·
`teapot_3d_load.doc` · `2DdataLoadSheet.doc` · `Synthetic.doc` (synthetic seismogram) ·
`ReservoirData 2005.xls` · `TeapotDomeBasemap.pdf` · `2DSeismicBasemap.pdf` ·
`NPR-3_TypeLog.jpg`

## Not downloaded

Available from `http://s3.amazonaws.com/teapot/` but deliberately skipped:

| File | Size | Reason |
|---|---|---|
| `npr3_gathers.sgy` | 5.69 GB | CDP gathers, statics, no NMO, floating datum (33,594 live of 64,860) |
| `npr3_field.sgy` | 5.69 GB | Raw field data with geometry |
| `npr3_dmo.vel` | 28 KB | DMO velocities — **worth grabbing**, the only real depth-conversion input |

`filt_mig.sgy` is also offered standalone there, but it is byte-identical (404,989,440 B)
to the copy inside `rmotc.tar` — no need to download it.

## Comparison with the other surveys

| | Teapot | Stratton | F3 |
|---|---|---|---|
| Domain | TWT 0–3000 ms | TWT 0–3000 ms | TWT 0–1848 ms |
| Grid | 345 × 188 × 1501 | 309 × 230 × 1501 | 651 × 951 × 462 |
| Missing traces | **0** | **0** | 3% (IL 701–750) |
| Cube size | 0.39 GB | 0.43 GB | 1.14 GB |
| Wells | 1,317 (deviated) | 21 (vertical) | 4 (vertical) |
| Sonic (DT) | in some LAS | **none** | all 4 |
| Tops carry TWT | no (MD only) | **yes** | no (checkshot needed) |
| Units | **feet** | feet | metres |

Smallest and densest volume of the three, and by far the most wells — but the weakest
well-tie metadata, since tops are MD-only and the wells are deviated.
