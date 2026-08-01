# Methodology

How `platescan` turns a dashcam clip into a list of plate sightings. This
documents the pipeline only — see `README.md` for installation and usage.

```
clip.MP4
  │
  ├─ ffprobe ──────────────► dimensions, frame rate, duration
  ├─ file name ────────────► clip start, camera, sequence
  ├─ moov/gps  box ────────► per-second UTC fixes, position, speed, bearing
  │
  └─ ffmpeg ───────────────► sampled ROI frames (raw, streamed)
                               │
                               ├─ 1. detect  ─► plate boxes, native resolution
                               ├─ 2. recognise ─► text + confidence per box
                               ├─ 3. accept  ─► plate-shaped readings only
                               └─ 4. still   ─► JPEG cut from this same frame
                                                 │
                                    detections ──┘
                                        │
                                        ├─ 5. track   ─► group into vehicles
                                        ├─ 6. coalesce ─► reunite split tracks
                                        └─ 7. report  ─► Markdown + JSON
```

Stages 1–4 run per frame across a worker pool. Stages 5–7 run once, over the
pooled detections.

---

## 0. Clip metadata

**File name.** `2026_0801_131218_000289F` is parsed as `YYYY_MMDD_HHMMSS_NNNNNN`
plus a camera letter (`F` front, `R` rear). This yields the clip's nominal start
time, its sequence number and which camera recorded it. An unrecognised name is
not fatal; the clip is simply reported without those fields.

**ffprobe** supplies width, height, frame rate and duration. Stream duration is
preferred, falling back to container duration.

## 1. GPS track

Novatek-based cameras store GPS outside any stream, so `ffprobe` does not see
it. The MP4 box tree is walked directly for `moov/gps ` — a **direct child of
`moov`, a sibling of `udta`, not inside it.**

The box payload is an 8-byte header (version, then a big-endian record count)
followed by that many big-endian `(file offset, size)` pairs, one per second of
video. Each offset points at a 16 KB region of `mdat` beginning with a 4-byte
length and the ASCII magic `freeGPS `.

Within a record, the numeric fields are **little-endian**, in contrast to the
big-endian boxes containing them:

| Offset | Type | Meaning |
|---|---|---|
| 4 | 8 bytes | magic `freeGPS ` |
| 48 | u32 ×6 | hour, minute, second, year (+2000), month, day — UTC |
| 72 | u8 ×3 | fix status `A`/`V`, latitude hemisphere, longitude hemisphere |
| 76 | f32 ×2 | latitude, longitude as NMEA `ddmm.mmmm` |
| 84 | f32 ×2 | speed in knots, bearing in degrees |

Coordinates are converted out of NMEA's degrees-and-minutes packing
(`deg = trunc(raw/100); deg + (raw - deg*100)/60`) and signed by hemisphere.
Records whose status is not `A` are discarded — the receiver had no lock. Index
*i* corresponds to clip second *i*.

**Time-zone inference.** The fixes are UTC; the file name is local. Comparing
the first valid fix against the file name, and rounding to the nearest quarter
hour, gives the camera's UTC offset. The rounding absorbs RTC drift, since every
real zone is a quarter-hour multiple. The residual after rounding is reported as
the camera's clock skew. **The fix's own index matters**: a receiver that starts
without a lock can leave the opening minutes empty, so the first fix is not
necessarily at second zero.

Timestamps then come from the satellites, falling back to file-name arithmetic
for stretches with no fix. Verified against this footage: GPS position matched
the camera's burned-in overlay exactly, while its clock ran 3 s ahead of it.

## 2. Frame sampling

One `ffmpeg` process per clip streams raw frames on stdout:

```
ffmpeg -ss <start> -i <clip> [-t <dur>] -an -sn -dn \
       -vf fps=<rate>,crop=<w>:<h>:<x>:<y> \
       -f rawvideo -pix_fmt <gray|rgb24> -
```

Cropping to the region of interest inside ffmpeg means only the useful band
crosses the pipe. Frames are read one at a time, so a 4 GB clip never needs to
be resident. Frame *n* is at clip offset `start + n/rate`.

The default ROI (`0.05,0.44,0.90,0.25`) is the road band of a forward dashcam:
below the horizon, above the bonnet, and clear of the burned-in overlay strip
along the bottom — which contains the camera's own vehicle tag and would
otherwise be "found" as a plate in every frame.

Pixel format follows the backend: grayscale for OCR (a third of the bytes),
RGB for the ONNX models.

## 3. Detection

The frame is covered with overlapping windows the size of the detector's input,
read from the model itself (640×640 for `yolo-v9-t-640`). **Windows are taken at
native resolution** — a plate 40 px wide in the source is still 40 px wide when
the model sees it. Downscaling a whole 4K frame to 640 would shrink it to ~7 px
and destroy it.

Each window is letterboxed into the square input, preserving aspect and padding
with YOLO's grey (114). A window that already fits is used at scale 1.0. Pixels
are converted to planar CHW `f32` scaled to 0..1.

Output is `[N, 7]` rows of `[batch, x1, y1, x2, y2, class, score]` — NMS is
already inside the exported graph. Rows below `--det-conf` are dropped, as are
boxes shorter than `--min-height` in source pixels. Surviving boxes are mapped
back through the letterbox (`(v - pad) / scale`), then by window origin, giving
region-of-interest coordinates.

Overlapping windows see the same plate more than once, so a final
**non-maximum suppression** pass over the whole frame discards any box with
IoU > 0.4 against a higher-scoring one.

## 4. Recognition

Every surviving box is cut from the frame at native resolution, resized to the
recognizer's input (128×64) and stacked into **one batched forward pass per
frame**. The tensor is `uint8` NHWC — rescaling lives inside the graph.

The `plate` head emits `[batch, slots, classes]`, already softmaxed. Each slot
takes its highest-scoring character from the alphabet
(`0-9A-Z` plus `_` as padding); padding slots are dropped, and the remainder
concatenated. **Confidence is the mean probability over the non-padding slots**,
rescaled to 0–100 to match the OCR backend's scale. A `region` head, when
present, gives the issuing country by argmax.

Alphabet, slot count and input geometry are read from the recognizer's sidecar
YAML, and the slot count is taken from the model's own output shape, so a
different recognizer can be dropped in without a rebuild.

## 5. Acceptance

Recognised text is uppercased and stripped to `A-Z0-9`, then must:

- be 4–8 characters,
- not be a known piece of road signage (`STOP`, `EXIT`, `SPEED`, …),
- contain at least 3 distinct characters — a run of one or two repeats is a
  texture artefact, not a plate.

Under `--strict-format` (implied for the OCR backend) it must also mix letters
with digits and match a regional format pattern. The ONNX backend is lenient by
default: its detector has already established that the box *is* a plate, so
demanding a national format there only discards valid vanity and out-of-region
plates.

## 6. Verification stills

A padded JPEG (2.4× wide, 3× tall, minimum 240×120) is cut around each accepted
box **from the frame that was analysed**, and carried with the detection.

This is deliberate. Re-decoding the clip afterwards with a seek to the same
timestamp lands on a neighbouring frame, and at road speeds a nearby vehicle
moves far enough in that gap to put the box on its tail light instead of its
plate. Measured on this footage: a seek to the recorded offset produced a frame
in the correct second but visibly displaced, ~165 px of vehicle motion.

## 7. Tracking

Detections from all workers are pooled and sorted by time, then assigned to
tracks in one pass.

**Character folding.** Before comparing, both spellings are folded over the
substitutions OCR makes systematically: `O D Q 0`→`0`, `I L 1`→`1`, `Z 2`→`2`,
`S 5`→`5`, `G 6`→`6`, `B 8`→`8`, `A 4`→`4`. This removes most disagreement
without merging genuinely different plates.

**Edit distance.** Whatever survives folding is allowed a Levenshtein distance
of 1, or 2 for plates of 7 characters or more.

**Assignment.** Each detection joins the open track with the smallest distance
within budget, or opens a new one. A track's match key follows its
highest-confidence reading, so a weak early guess does not anchor it.

**The same-frame rule.** A detection never joins a track that already has a
reading in that frame, and two such tracks are never merged later. Two plates
read in one frame are two vehicles, however alike they look — this is what stops
cars sitting side by side in traffic from collapsing into one row.

**Dropout window.** A track closes after a quiet period. The base figure
(`--gap`) is scaled by the camera vehicle's own GPS speed, because how long
another vehicle stays in view depends entirely on how fast you are moving:
crawling in traffic the car ahead is still there thirty seconds later, at 60 mph
you have passed it. The factor is `30 mph / ego speed`, clamped to 0.5×–4×.
Without GPS the base figure is used unchanged.

**Coalescing.** Nearest-track assignment cannot undo an early split: once a bad
reading has opened its own track, later good readings join whichever is closer
and the two never reunite. A pass afterwards repeatedly merges any two tracks
that pass the same-frame rule, are within edit distance, and are close enough in
time. Spellings that agree *exactly* are trusted across twice the window of a
pair that had to be guessed at, since an exact match is strong evidence of one
vehicle.

**Sighting statistics.** A merged track reports: the spelling carrying the most
total confidence (not the most frequent), first and last seen, the count of
*distinct frames* (so two windows reading one plate in a single frame cannot
satisfy `--min-hits`), mean and best confidence, and every distinct spelling
with its count. Tracks below `--min-hits` distinct frames are discarded.

## 8. Output

Markdown, plus JSON when asked. Each sighting carries both a clip offset and a
wall clock, the GPS time and position where available, the detector score, the
predicted region, the source-frame box, all OCR variants, and a link to its
still.

Reruns never overwrite: the report is written beside any existing one as
`-2`, `-3`, … unless `--force` is given.

---

## Coordinate systems

Three, and confusing them is the easiest way to produce plausible nonsense:

| Space | Origin | Used by |
|---|---|---|
| **Window** | top-left of a detector window | raw model output |
| **Region of interest** | top-left of the ffmpeg crop | frame pixel access, recognition crops |
| **Source frame** | top-left of the full video frame | everything reported |

Detector output is unpadded and unscaled into window space, shifted by the
window origin into ROI space, and shifted by the crop origin into source space
for reporting. Frame pixel lookups always convert back to ROI space first.

## Known failure modes

- **Legibility, not detection, is the limit.** Plates are detected further away
  than they can be read. A larger detector finds more boxes and produces more
  unreadable fragments, not more vehicles.
- **A wrong reading looks exactly like a right one.** Confidence tracks
  legibility closely and is the useful filter, but nothing in the pipeline can
  tell a confident misread from a correct read. The still is the check.
- **Intermittent detection still splits vehicles**, typically once far away and
  once close, when the two spellings differ too much to reunite.
- **Absence is not evidence.** Plates outside the ROI, too small, too angled or
  motion-blurred never enter the pipeline at all.
