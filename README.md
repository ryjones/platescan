# platescan

Scans dashcam video for license plates and writes a timestamped Markdown report.

```
platescan 2026_0801_131218_000289F.MP4
```

Produces `2026_0801_131218_000289F-plates.md` plus a `-crops/` directory holding
one still per sighting, so every reading can be checked by eye.

## What it does

1. Decodes the clip's start, camera and sequence from the file name
   (`2026_0801_131218_000289F` -> 2026-08-01 13:12:18, clip 289, front camera),
   and reads the embedded GPS track for satellite time and position.
2. Streams frames out of ffmpeg at a sample rate you choose, cropped to a
   region of interest.
3. Runs a plate detector over each frame in overlapping windows **at native
   resolution**, so a plate 40 px wide in a 4K frame is still 40 px wide when
   the model sees it. Duplicate boxes from overlapping windows are merged.
4. Feeds each detected plate to a recognizer trained on plates, batched per
   frame.
5. Groups repeat readings into sightings, so one vehicle is one row. See
   *Merging* below.
6. Saves a verification still for each sighting, encoded from the frame that
   was analysed rather than re-seeking the clip afterwards — a seek lands on a
   neighbouring frame, and at road speeds that is enough to put the box on a
   tail light instead of the plate.
7. Writes Markdown (and optionally JSON) with first/last seen times as both a
   clip offset and a wall clock.

## Merging

A plate read over twenty frames comes back as a handful of near-identical
strings, so grouping them is most of the work of not reporting one car three
times. Four things decide whether two readings are the same vehicle:

- **Character folding.** The confusions OCR makes systematically (`0`/`O`,
  `1`/`I`, `5`/`S`, `8`/`B`, `4`/`A`, `6`/`G`) are collapsed before comparing.
- **Edit distance.** Whatever difference survives folding is allowed up to one
  character, or two on plates of seven characters or more.
- **A second pass.** Assigning each reading to its nearest open track cannot
  undo an early split: once a bad reading has opened its own track, later good
  readings join whichever is closer and the two never reunite. A pass afterwards
  reunites them, and the confident spelling wins the merged row.
- **Ego speed, from GPS.** How long a vehicle stays in view depends entirely on
  how fast you are going. Crawling in traffic, the car ahead is still there
  thirty seconds later; at 60 mph you have passed it. The dropout allowed before
  a sighting is closed is scaled against speed, four times wider at a standstill
  and half as wide at speed. Readings that agree exactly are trusted across
  twice the gap of readings that had to be guessed at.

The one rule that overrides all of it: **two plates read in the same frame are
two different vehicles**, however alike they look. That is what stops the merge
from collapsing cars sitting side by side in traffic.

Sightings can still split when a vehicle is detected intermittently, typically
once far away and once close. Raise `--gap` to merge harder.

## Timestamps and GPS

Novatek-based dashcams (Viofo and similar) embed a GPS track in the MP4: a
`moov/gps ` index pointing at one `freeGPS` record per second of video, each
carrying a UTC fix, position, speed and bearing. `platescan` reads it directly,
with no exiftool dependency.

That matters because the file name only carries the camera's own RTC, which
drifts — on this footage it ran 2-3 seconds off satellite time, and the
burned-in overlay drifts with it. When a GPS track is present, `platescan` uses
satellite time for every sighting, infers the local UTC offset by comparing the
first fix against the file name (rounded to the nearest quarter hour), and
reports the measured skew so you can see how far the camera's clock was out.

Each sighting then also carries where it happened:

```
- GPS time **2026-08-01 19:03:13.000Z**
- Seen at [47.64834, -122.18768](https://www.openstreetmap.org/?mlat=...),
  travelling 31 mph (51 km/h) on a bearing of 182°
```

A receiver that starts without a lock leaves the opening minutes without fixes;
those stretches fall back to the file-name clock, and the report says which
clock it used. Rear-camera clips carry no GPS of their own, so `platescan`
borrows the track from the paired front clip — one scanned in the same run, or
failing that one found beside the file with an overlapping time stamp — shifted
into the rear clip's timeline. The report marks the track as borrowed.

## Google Earth

`--kmz` writes a KMZ alongside the report: one placemark per sighting at the
GPS fix of its best reading, with the verification crop embedded in the
balloon, a timestamp for Earth's time slider, and each clip's drive traced as
a line.

```
platescan clip.MP4 --kmz clip.kmz
```

A KMZ can also be built from a previous run without rescanning, by passing
that run's `--json` report as the input; crops are picked up from beside the
report:

```
platescan reports/clip-plates.json --kmz clip.kmz     # --kmz optional here
```

Sightings with no fix (no GPS and no borrowable pair) are left off the map,
with a warning saying how many.

## Trips

`--by-trip` turns a pile of clips into one consolidated report per drive:

```
platescan /footage/2026_0727*.MP4 --by-trip --out reports --kmz
```

A directory input is scanned recursively for MP4s and implies `--by-trip`,
so months of footage can be handed over as-is; duplicate stems (locked-clip
backup folders) are scanned once. Clips are sorted by wall-clock start and
split wherever the camera was off longer than `--trip-gap` (90 s by
default); front and rear clips recorded together always land in the same
trip. Each trip gets
`<start>-trip-plates.md`, a JSON of raw findings (always written in this
mode, so a KMZ can be rebuilt later without rescanning), crops, and — with
`--kmz` — a Google Earth file, all named after the trip's start time. In
this mode `--out` names a directory.

Clips already covered by any platescan JSON in the output directory are not
rescanned: their findings are folded straight into the trip report, crops
and all. Consolidating a directory of per-clip reports into trip reports
therefore only pays for clips that were never scanned.

## Requirements

```
brew install ffmpeg onnxruntime
brew install tesseract          # only for --engine tesseract
```

Then fetch the models (MIT licensed, ~13 MB total):

```
mkdir -p models && cd models
curl -LO https://github.com/ankandrew/open-image-models/releases/download/assets/yolo-v9-t-640-license-plates-end2end.onnx
for f in cct_s_v2_global.onnx cct_s_v2_global_plate_config.yaml; do
  curl -LO https://github.com/ankandrew/fast-plate-ocr/releases/download/arg-plates/$f
done
```

`platescan` picks the detector and recognizer out of `--models` by file name, so
you can swap in any of the other sizes from those releases. The recognizer's
alphabet, input size and region list are read from the sidecar YAML, so a
different recognizer works without a rebuild.

## Engines

| | `--engine alpr` (default) | `--engine tesseract` |
|---|---|---|
| Approach | plate detector + plate recognizer | tiled general-purpose OCR |
| Reads plates at road distance | yes | no |
| Extra setup | onnxruntime + models | tesseract |
| Speed on 4K | ~0.7 s per sampled frame | ~1.5 s per sampled frame |

The tesseract engine is kept as a fallback for machines without onnxruntime. It
only reads plates that are large in frame — measured on this footage, a vehicle
a few car lengths ahead has a plate around 45-90 px wide with ~12 px character
height, and tesseract read nothing at that size at any upscale factor or
preprocessing chain tried. The ALPR engine reads those same plates correctly.

## Accuracy — read this before trusting the output

Readings are model output, not verified plates. **Check the crop before relying
on any reading.** The report links one for every sighting.

Confidence tracks legibility closely and is the main thing to filter on. On a
sample of dense freeway traffic, sightings at confidence 100 were correct on
inspection, while those in the 60s were plates too distant to read by eye
either. Raise `--min-conf` when you want only the solid ones.

Known limits:

- A vehicle detected intermittently can still appear as more than one sighting.
  See *Merging*.
- Times come from GPS when the clip carries a track, and otherwise from the
  file name, which is only as good as the camera's RTC. The report states which.
- Angled, motion-blurred, dirty and partly occluded plates read worse, and the
  failure is usually a plausible-looking wrong character rather than a blank.
- Absence from the report is not evidence a vehicle was absent.

## Reruns never overwrite

A scan is expensive and its output is evidence, so rerunning writes
`report-2.md`, `report-3.md`, and so on beside the existing one, each with its
own crops directory. Pass `--force` to overwrite deliberately.

## Choosing models

The detector ships in several sizes. Bigger is not better here: on a minute of
dense freeway traffic, `yolo-v9-s-608` (28.6 MB) took **3.1x** the runtime of
`yolo-v9-t-640` (7.8 MB) and turned 14 sightings into 18 — but the four extra
were not four extra cars. They were fragments of vehicles already found
(`FP07114`, `FV1114`, `RP70318`, `70319` are all the same Lexus as `CPZ0319`),
because the larger detector finds plates at distances where the recognizer
cannot read them. Detection is not the bottleneck; legibility is. Stay on the
tiny detector and spend the time on `--fps` instead.

## The on-screen display

Dashcams burn a strip along the bottom of the frame: timestamp, speed, GPS, and
the owner's own vehicle tag. That tag is plate-shaped and will be "found" in
every frame. The default region of interest excludes the bottom of the frame;
`platescan` warns if you widen `--roi` back over it.

## Options

| Flag | Default | Meaning |
|---|---|---|
| `--engine` | `alpr` | `alpr` or `tesseract` |
| `--models` | `models` | Directory holding the ONNX detector and recognizer |
| `--ort-dylib` | auto | Path to libonnxruntime |
| `--det-conf` | `0.35` | Minimum plate-detector objectness (alpr) |
| `--fps` | `2.0` | Frames sampled per second of video |
| `--start` / `--end` | whole clip | Time window in seconds |
| `--roi x,y,w,h` | `0.05,0.44,0.90,0.25` | Region of interest as frame fractions; the default is the road band below the horizon and above the bonnet |
| `--min-conf` | `60` | Confidence floor, 0-100 for either engine |
| `--min-height` | `9` | Minimum plate height in source pixels |
| `--min-hits` | `2` | Distinct frames a plate must appear in |
| `--gap` | `10.0` | Base dropout before a sighting is closed; scaled by GPS speed |
| `--region` | `generic` | Plate format preset: `generic`, `us`, `eu` |
| `--pattern` | - | Custom plate regex; repeatable, replaces the preset |
| `--strict-format` | off | Enforce the format preset on ALPR output too |
| `--overlap` | `0.25` | Overlap between neighbouring windows |
| `--tile WxH` | `960x540` | OCR tile size (tesseract) |
| `--upscale` | `3.0` | Upscale before OCR (tesseract) |
| `--psm` | `11` | Page segmentation mode (tesseract) |
| `-j`, `--jobs` | cores - 1 | Parallel workers |
| `--json` | - | Also write raw findings as JSON |
| `--kmz` | - | Also write a Google Earth KMZ with embedded crops; takes a `--json` report as input to skip rescanning |
| `--by-trip` | off | One consolidated report per contiguous run of clips; `--out` becomes a directory |
| `--trip-gap` | `90` | Seconds of camera-off that still counts as the same trip |
| `--no-crops` | off | Skip saving verification stills |
| `--force` | off | Overwrite an existing report instead of writing beside it |

Multiple inputs are combined into one report with a section per clip.

## Tuning

Missing plates? Sample more often and lower the detector threshold:

```
platescan clip.MP4 --fps 4 --det-conf 0.25 --min-conf 45
```

Too much noise? Raise `--min-conf`, raise `--min-hits`, or pin the format with
`--region us --strict-format`.

Runtime scales with sampled frames x detector windows. Widening `--roi`
vertically past one window height doubles the work, so keep it to the band where
plates actually appear.

## Layout

`METHODOLOGY.md` documents the pipeline itself — the formats, transforms and
decision rules at each stage.

| File | Role |
|---|---|
| `src/cli.rs` | Argument parsing and validation |
| `src/clip.rs` | File-name timestamp decoding, ffprobe metadata, clock selection |
| `src/gps.rs` | MP4 box walking, `freeGPS` record parsing, time-zone inference |
| `src/frames.rs` | ffmpeg frame streaming, cropping, still extraction |
| `src/scan.rs` | `Scanner` backend trait, candidates, window tiling |
| `src/alpr.rs` | ONNX detector + recognizer, letterboxing, NMS, decoding |
| `src/ocr.rs` | Tesseract backend: tiling, upscaling, TSV parsing |
| `src/plate.rs` | Plate format rules and character folding |
| `src/track.rs` | Grouping detections into sightings |
| `src/report.rs` | Markdown and JSON output |

`cargo test` covers name parsing, window coverage, TSV parsing and word joining,
plate acceptance, NMS and letterboxing, slot decoding, sighting grouping, and
GPS coordinate/time-zone maths including a late first lock.

`cargo run --release --example probe -- models/*.onnx` prints model signatures;
`--example alpr_test -- frame.png` runs the pair against a single still.
