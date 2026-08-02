//! Google Earth export.
//!
//! Writes a KMZ rather than bare KML because KMZ is the only form Google
//! Earth accepts with images embedded: a zip holding `doc.kml` plus the
//! verification crops under `files/`. Each sighting becomes a placemark at
//! the GPS fix of its best reading, with the crop in the balloon, and each
//! clip's drive is traced as a line.

use std::fmt::Write as _;
use std::fs::File;
use std::io::Write as _;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use chrono::NaiveDateTime;
use serde_json::Value;
use zip::write::SimpleFileOptions;
use zip::{CompressionMethod, ZipWriter};

use crate::report::{offset, ClipReport};

pub struct KmlClip {
    pub stem: String,
    /// Camera label for the folder name, when known.
    pub camera: Option<String>,
    /// lon, lat pairs tracing the drive.
    pub route: Vec<(f64, f64)>,
    pub placemarks: Vec<Placemark>,
    /// Sightings left off the map because no fix covers them.
    pub unplaced: usize,
}

pub struct Placemark {
    pub plate: String,
    pub lat: f64,
    pub lon: f64,
    /// ISO-8601 UTC instant for Earth's time slider.
    pub when_utc: Option<String>,
    pub first_wall: Option<String>,
    pub first_offset: String,
    pub visible_s: f64,
    pub frames: usize,
    pub best_conf: f32,
    pub mean_conf: f32,
    pub detector_score: Option<f64>,
    pub speed_mph: Option<f64>,
    pub bearing: Option<f64>,
    pub variants: Vec<(String, usize)>,
    /// Archive file name and JPEG bytes of the verification crop.
    pub image: Option<(String, Vec<u8>)>,
}

/// Build the export straight from a scan's in-memory reports.
pub fn from_reports(reports: &[ClipReport]) -> Vec<KmlClip> {
    reports
        .iter()
        .map(|r| {
            let c = &r.clip;
            let route = c
                .gps
                .as_ref()
                .map(|t| t.points().map(|f| (f.lon, f.lat)).collect())
                .unwrap_or_default();
            let mut unplaced = 0;
            let placemarks = r
                .sightings
                .iter()
                .enumerate()
                .filter_map(|(i, s)| {
                    let Some(fix) = c.fix_at(s.best_offset) else {
                        unplaced += 1;
                        return None;
                    };
                    Some(Placemark {
                        plate: s.plate.clone(),
                        lat: fix.lat,
                        lon: fix.lon,
                        when_utc: c
                            .gps_utc(s.best_offset)
                            .or_else(|| c.gps_utc(s.first_seen))
                            .map(|t| t.format("%Y-%m-%dT%H:%M:%SZ").to_string()),
                        first_wall: c
                            .wall_clock(s.first_seen)
                            .map(|t| t.format("%Y-%m-%d %H:%M:%S%.3f").to_string()),
                        first_offset: offset(s.first_seen),
                        visible_s: s.duration(),
                        frames: s.frames,
                        best_conf: s.best_conf,
                        mean_conf: s.mean_conf,
                        detector_score: s.detector_score.map(f64::from),
                        speed_mph: Some(fix.speed_mph() as f64),
                        bearing: Some(fix.bearing as f64),
                        variants: s.variants.clone(),
                        image: s
                            .still
                            .as_ref()
                            .map(|bytes| (s.crop_name(&c.stem, i), bytes.clone())),
                    })
                })
                .collect();
            KmlClip {
                stem: c.stem.clone(),
                camera: c.camera.map(|cam| cam.label().to_string()),
                route,
                placemarks,
                unplaced,
            }
        })
        .collect()
}

/// Build the export from a previous run's `--json` report, so a KMZ can be
/// made without rescanning. Crop paths in the JSON are resolved relative to
/// the report; the route comes from the source clip's GPS track when the
/// clip is still reachable.
pub fn from_json(path: &Path) -> Result<Vec<KmlClip>> {
    let text = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    let doc: Value = serde_json::from_str(&text)
        .with_context(|| format!("{} is not valid JSON", path.display()))?;
    let Some(clips) = doc["clips"].as_array() else {
        bail!(
            "{}: not a platescan report (no \"clips\" array; was it written with --json?)",
            path.display()
        );
    };
    let base = path.parent().unwrap_or(Path::new(""));
    clips.iter().map(|c| json_clip(c, base)).collect()
}

fn json_clip(c: &Value, base: &Path) -> Result<KmlClip> {
    let stem = c["stem"].as_str().unwrap_or("clip").to_string();
    let start = c["clip_start"]
        .as_str()
        .and_then(|s| NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S").ok());
    let duration = c["duration_s"].as_f64().unwrap_or(600.0);
    // The JSON carries each sighting's own fix; only the route needs the
    // source clip, and only if it is still where the report says.
    let track = c["file"]
        .as_str()
        .map(Path::new)
        .filter(|p| p.exists())
        .and_then(|p| crate::clip::disk_track(p, start, duration));
    let route = track
        .as_ref()
        .map(|t| t.points().map(|f| (f.lon, f.lat)).collect())
        .unwrap_or_default();

    let mut unplaced = 0;
    let mut placemarks = Vec::new();
    for s in c["sightings"].as_array().map(Vec::as_slice).unwrap_or(&[]) {
        let best_offset = s["best_offset_s"].as_f64().unwrap_or(0.0);
        // Prefer the fix recorded in the report; fall back to the track for
        // reports written before positions were recorded, or rear clips
        // whose pair is only reachable now.
        let (lat, lon, speed_mph, bearing) = match (
            s["gps"]["lat"].as_f64(),
            s["gps"]["lon"].as_f64(),
        ) {
            (Some(lat), Some(lon)) => (
                lat,
                lon,
                s["gps"]["speed_mph"].as_f64(),
                s["gps"]["bearing_deg"].as_f64(),
            ),
            _ => match track.as_ref().and_then(|t| t.nearest(best_offset)) {
                Some(fix) => (
                    fix.lat,
                    fix.lon,
                    Some(fix.speed_mph() as f64),
                    Some(fix.bearing as f64),
                ),
                None => {
                    unplaced += 1;
                    continue;
                }
            },
        };
        let when_utc = s["gps_utc"]
            .as_str()
            .and_then(|t| t.get(..19).map(|p| format!("{p}Z")))
            .or_else(|| {
                track
                    .as_ref()
                    .and_then(|t| t.nearest(best_offset))
                    .map(|f| f.time.format("%Y-%m-%dT%H:%M:%SZ").to_string())
            });
        let image = s["crop"].as_str().and_then(|rel| {
            let p = PathBuf::from(rel);
            let full = if p.is_absolute() { p } else { base.join(p) };
            let name = full.file_name()?.to_str()?.to_string();
            std::fs::read(&full).ok().map(|bytes| (name, bytes))
        });
        let first_seen = s["first_seen_s"].as_f64().unwrap_or(0.0);
        placemarks.push(Placemark {
            plate: s["plate"].as_str().unwrap_or("?").to_string(),
            lat,
            lon,
            when_utc,
            first_wall: s["first_seen_wall"]
                .as_str()
                .map(|t| t.replacen('T', " ", 1)),
            first_offset: offset(first_seen),
            visible_s: s["last_seen_s"].as_f64().unwrap_or(first_seen) - first_seen,
            frames: s["frames"].as_u64().unwrap_or(0) as usize,
            best_conf: s["best_conf"].as_f64().unwrap_or(0.0) as f32,
            mean_conf: s["mean_conf"].as_f64().unwrap_or(0.0) as f32,
            detector_score: s["detector_score"].as_f64(),
            speed_mph,
            bearing,
            variants: s["variants"]
                .as_array()
                .map(Vec::as_slice)
                .unwrap_or(&[])
                .iter()
                .filter_map(|v| {
                    Some((
                        v["text"].as_str()?.to_string(),
                        v["count"].as_u64()? as usize,
                    ))
                })
                .collect(),
            image,
        });
    }
    Ok(KmlClip {
        stem,
        camera: c["camera"].as_str().map(str::to_string),
        route,
        placemarks,
        unplaced,
    })
}

pub fn write_kmz(path: &Path, clips: &[KmlClip]) -> Result<()> {
    let kml = document(clips);
    let file =
        File::create(path).with_context(|| format!("failed to write {}", path.display()))?;
    let mut zip = ZipWriter::new(file);
    // JPEG does not compress further; storing keeps the zip dependency to
    // its core and Google Earth reads either.
    let opts = SimpleFileOptions::default().compression_method(CompressionMethod::Stored);
    zip.start_file("doc.kml", opts)?;
    zip.write_all(kml.as_bytes())?;
    for clip in clips {
        for p in &clip.placemarks {
            if let Some((name, bytes)) = &p.image {
                zip.start_file(format!("files/{name}"), opts)?;
                zip.write_all(bytes)?;
            }
        }
    }
    zip.finish()
        .with_context(|| format!("failed to finish {}", path.display()))?;
    Ok(())
}

fn document(clips: &[KmlClip]) -> String {
    let mut k = String::new();
    let title = match clips {
        [only] => format!("License plate scan — {}", only.stem),
        many => format!("License plate scan — {} clips", many.len()),
    };
    let w = &mut k;
    let _ = writeln!(w, r#"<?xml version="1.0" encoding="UTF-8"?>"#);
    let _ = writeln!(w, r#"<kml xmlns="http://www.opengis.net/kml/2.2">"#);
    let _ = writeln!(w, "<Document>");
    let _ = writeln!(w, "<name>{}</name>", esc(&title));
    let _ = writeln!(
        w,
        "<Style id=\"plate\"><IconStyle><scale>1.1</scale><Icon>\
         <href>http://maps.google.com/mapfiles/kml/shapes/cabs.png</href>\
         </Icon></IconStyle><LabelStyle><scale>0.8</scale></LabelStyle></Style>"
    );
    let _ = writeln!(
        w,
        "<Style id=\"route\"><LineStyle><color>7fff5500</color>\
         <width>3</width></LineStyle></Style>"
    );
    for clip in clips {
        let folder = match &clip.camera {
            Some(cam) => format!("{} — {cam} camera", clip.stem),
            None => clip.stem.clone(),
        };
        let _ = writeln!(w, "<Folder><name>{}</name>", esc(&folder));
        if clip.route.len() > 1 {
            let _ = write!(
                w,
                "<Placemark><name>Route</name><styleUrl>#route</styleUrl>\
                 <LineString><tessellate>1</tessellate><coordinates>"
            );
            for (lon, lat) in &clip.route {
                let _ = write!(w, "{lon:.6},{lat:.6},0 ");
            }
            let _ = writeln!(w, "</coordinates></LineString></Placemark>");
        }
        for p in &clip.placemarks {
            let _ = writeln!(w, "<Placemark>");
            let _ = writeln!(w, "<name>{}</name>", esc(&p.plate));
            let _ = writeln!(w, "<styleUrl>#plate</styleUrl>");
            if let Some(when) = &p.when_utc {
                let _ = writeln!(w, "<TimeStamp><when>{}</when></TimeStamp>", esc(when));
            }
            let _ = writeln!(
                w,
                "<description><![CDATA[{}]]></description>",
                balloon(p, &folder)
            );
            let _ = writeln!(
                w,
                "<Point><coordinates>{:.6},{:.6},0</coordinates></Point>",
                p.lon, p.lat
            );
            let _ = writeln!(w, "</Placemark>");
        }
        let _ = writeln!(w, "</Folder>");
    }
    let _ = writeln!(w, "</Document>");
    let _ = writeln!(w, "</kml>");
    k
}

/// Balloon HTML. Google Earth renders it with WebKit, so plain HTML works;
/// the image reference resolves inside the KMZ.
fn balloon(p: &Placemark, folder: &str) -> String {
    let mut b = String::new();
    if let Some((name, _)) = &p.image {
        let _ = write!(b, r#"<img src="files/{name}" style="max-width:420px"/><br/>"#);
    }
    let _ = write!(b, "<b>{}</b> — {}<br/>", p.plate, folder);
    if let Some(wall) = &p.first_wall {
        let _ = write!(b, "First seen {wall} (offset {})", p.first_offset);
    } else {
        let _ = write!(b, "First seen at offset {}", p.first_offset);
    }
    let _ = write!(
        b,
        ", visible {:.1} s over {} frame(s)<br/>",
        p.visible_s, p.frames
    );
    let _ = write!(
        b,
        "Confidence best {:.0}, mean {:.0}",
        p.best_conf, p.mean_conf
    );
    if let Some(score) = p.detector_score {
        let _ = write!(b, "; detector {score:.2}");
    }
    let _ = write!(b, "<br/>");
    if let (Some(mph), Some(bearing)) = (p.speed_mph, p.bearing) {
        let _ = write!(
            b,
            "Travelling {mph:.0} mph on a bearing of {bearing:.0}°<br/>"
        );
    }
    if !p.variants.is_empty() {
        let variants = p
            .variants
            .iter()
            .map(|(t, n)| format!("{t} ×{n}"))
            .collect::<Vec<_>>()
            .join(", ");
        let _ = write!(b, "Readings: {variants}<br/>");
    }
    let _ = write!(
        b,
        "<i>Model output, not a verified plate; check the image.</i>"
    );
    b
}

fn esc(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mark(plate: &str, image: Option<(String, Vec<u8>)>) -> Placemark {
        Placemark {
            plate: plate.into(),
            lat: 47.5,
            lon: -122.2,
            when_utc: Some("2026-07-18T18:37:10Z".into()),
            first_wall: Some("2026-07-18 11:37:10.000".into()),
            first_offset: "00:07.000".into(),
            visible_s: 3.2,
            frames: 6,
            best_conf: 91.0,
            mean_conf: 84.0,
            detector_score: Some(0.88),
            speed_mph: Some(42.0),
            bearing: Some(213.0),
            variants: vec![("ABC123".into(), 4)],
            image,
        }
    }

    #[test]
    fn document_places_the_sighting() {
        let clip = KmlClip {
            stem: "2026_0718_183703_000033F".into(),
            camera: Some("front".into()),
            route: vec![(-122.2, 47.5), (-122.201, 47.501)],
            placemarks: vec![mark("ABC123", None)],
            unplaced: 0,
        };
        let doc = document(&[clip]);
        assert!(doc.contains("<name>ABC123</name>"));
        assert!(doc.contains("-122.200000,47.500000,0"));
        assert!(doc.contains("<when>2026-07-18T18:37:10Z</when>"));
        assert!(doc.contains("LineString"), "route line present");
    }

    #[test]
    fn kmz_is_a_zip_with_the_image_embedded() {
        let dir = std::env::temp_dir();
        let path = dir.join("platescan-kml-test.kmz");
        let clip = KmlClip {
            stem: "clip".into(),
            camera: None,
            route: Vec::new(),
            placemarks: vec![mark(
                "XYZ789",
                Some(("clip-001-XYZ789.jpg".into(), vec![0xFF, 0xD8, 0xFF, 0xD9])),
            )],
            unplaced: 0,
        };
        write_kmz(&path, &[clip]).expect("writes");
        let bytes = std::fs::read(&path).expect("readable");
        let _ = std::fs::remove_file(&path);
        assert_eq!(&bytes[..2], b"PK", "zip magic");
        let hay = bytes.windows(7).any(|win| win == b"doc.kml");
        assert!(hay, "doc.kml entry present");
        let img = bytes
            .windows(b"files/clip-001-XYZ789.jpg".len())
            .any(|win| win == b"files/clip-001-XYZ789.jpg");
        assert!(img, "image entry present");
    }

    #[test]
    fn xml_escapes_text_nodes() {
        assert_eq!(esc("A&W <cars>"), "A&amp;W &lt;cars&gt;");
    }
}
