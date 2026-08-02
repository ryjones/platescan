//! Self-contained interactive map report.
//!
//! One HTML file per report: a Leaflet map with each clip's route traced and
//! one marker per sighting, the verification crop embedded in the popup as a
//! base64 data URI. Exists because Google Earth's web and mobile viewers
//! refuse balloon images that are not public https URLs — they will not read
//! them out of a KMZ archive, and hosting plate crops publicly is exactly
//! what an evidence workflow should not do. A browser has no such rule: this
//! file works anywhere, and the only network it touches is the Leaflet
//! library and the OpenStreetMap base tiles; the plate data never leaves the
//! file.
//!
//! Builds from the same model as the KMZ writer, so everything that can make
//! a KMZ — a live scan, a trip, a previous run's JSON — can make this too.

use std::path::Path;

use anyhow::{Context, Result};
use base64::Engine as _;
use serde_json::json;

use crate::kml::KmlClip;

pub fn write_html(path: &Path, clips: &[KmlClip]) -> Result<()> {
    let doc = document(clips);
    std::fs::write(path, doc).with_context(|| format!("failed to write {}", path.display()))
}

fn document(clips: &[KmlClip]) -> String {
    let title = match clips {
        [only] => format!("License plate scan — {}", only.stem),
        many => format!("License plate scan — {} clips", many.len()),
    };

    // Routes as [lat, lon] pairs — Leaflet's order, opposite to KML's.
    let routes: Vec<_> = clips
        .iter()
        .filter(|c| c.route.len() > 1)
        .map(|c| {
            c.route
                .iter()
                .map(|(lon, lat)| json!([lat, lon]))
                .collect::<Vec<_>>()
        })
        .collect();

    let mut sightings = Vec::new();
    for clip in clips {
        let label = match &clip.camera {
            Some(cam) => format!("{} — {cam} camera", clip.stem),
            None => clip.stem.clone(),
        };
        for p in &clip.placemarks {
            let variants = p
                .variants
                .iter()
                .map(|(t, n)| format!("{t} ×{n}"))
                .collect::<Vec<_>>()
                .join(", ");
            sightings.push(json!({
                "plate": p.plate,
                "lat": p.lat,
                "lon": p.lon,
                "clip": label,
                "wall": p.first_wall,
                "offset": p.first_offset,
                "visible": p.visible_s,
                "frames": p.frames,
                "bestConf": p.best_conf,
                "meanConf": p.mean_conf,
                "det": p.detector_score,
                "mph": p.speed_mph,
                "bearing": p.bearing,
                "variants": variants,
                "img": p.image.as_ref().map(|(_, bytes)| format!(
                    "data:image/jpeg;base64,{}",
                    base64::engine::general_purpose::STANDARD.encode(bytes)
                )),
            }));
        }
    }
    let unplaced: usize = clips.iter().map(|c| c.unplaced).sum();

    // `</` cannot appear inside an inline <script>; the data is JSON-encoded
    // strings, so escaping the slash is enough and changes nothing else.
    let embed = |v: &serde_json::Value| {
        serde_json::to_string(v)
            .unwrap_or_else(|_| "[]".into())
            .replace("</", "<\\/")
    };
    let routes_js = embed(&json!(routes));
    let sightings_js = embed(&json!(sightings));

    let note = if unplaced > 0 {
        format!(
            " &middot; {unplaced} sighting(s) had no GPS fix and are not shown"
        )
    } else {
        String::new()
    };

    format!(
        r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
  html, body, #map {{ height: 100%; margin: 0; }}
  #hdr {{
    position: absolute; top: 10px; left: 50px; right: 10px; z-index: 1000;
    background: rgba(255,255,255,0.92); border-radius: 6px; padding: 6px 12px;
    font: 14px/1.4 system-ui, sans-serif; box-shadow: 0 1px 4px rgba(0,0,0,0.3);
    width: fit-content; max-width: calc(100% - 80px);
  }}
  .popup {{ font: 13px/1.45 system-ui, sans-serif; }}
  .popup img {{ max-width: 320px; display: block; margin-bottom: 6px; border-radius: 3px; }}
  .popup .plate {{ font-size: 1.25em; font-weight: 700; }}
</style>
</head>
<body>
<div id="map"></div>
<div id="hdr"><b>{title_esc}</b> &middot; {count} sighting(s){note}</div>
<script>
const ROUTES = {routes_js};
const SIGHTINGS = {sightings_js};

const map = L.map('map');
L.tileLayer('https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
  maxZoom: 19,
  attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
}}).addTo(map);

function esc(t) {{
  return String(t).replace(/[&<>"]/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}})[c]);
}}

function popup(s) {{
  let h = '<div class="popup">';
  if (s.img) h += '<img src="' + s.img + '" alt="' + esc(s.plate) + '">';
  h += '<span class="plate">' + esc(s.plate) + '</span> &mdash; ' + esc(s.clip) + '<br>';
  h += 'First seen ' + (s.wall ? esc(s.wall) : 'at offset ' + esc(s.offset))
     + ' (offset ' + esc(s.offset) + '), visible ' + s.visible.toFixed(1)
     + ' s over ' + s.frames + ' frame(s)<br>';
  h += 'Confidence best ' + Math.round(s.bestConf) + ', mean ' + Math.round(s.meanConf);
  if (s.det != null) h += '; detector ' + s.det.toFixed(2);
  h += '<br>';
  if (s.mph != null) h += 'Travelling ' + Math.round(s.mph) + ' mph on a bearing of '
     + Math.round(s.bearing) + '&deg;<br>';
  if (s.variants) h += 'Readings: ' + esc(s.variants) + '<br>';
  h += '<i>Model output, not a verified plate; check the image.</i></div>';
  return h;
}}

const bounds = [];
for (const r of ROUTES) {{
  L.polyline(r, {{ color: '#ff7f2a', weight: 3, opacity: 0.75 }}).addTo(map);
  for (const p of r) bounds.push(p);
}}
for (const s of SIGHTINGS) {{
  const m = L.marker([s.lat, s.lon]).addTo(map);
  m.bindTooltip(s.plate);
  m.bindPopup(popup(s), {{ maxWidth: 360 }});
  bounds.push([s.lat, s.lon]);
}}
if (bounds.length) map.fitBounds(bounds, {{ padding: [40, 40] }});
else map.setView([0, 0], 2);
</script>
</body>
</html>
"#,
        title = esc(&title),
        title_esc = esc(&title),
        count = sightings.len(),
    )
}

fn esc(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kml::Placemark;

    #[test]
    fn document_embeds_route_marker_and_image() {
        let clip = KmlClip {
            stem: "2026_0717_113820_000003F".into(),
            camera: Some("front".into()),
            route: vec![(-122.2, 47.5), (-122.201, 47.501)],
            placemarks: vec![Placemark {
                plate: "ABC123".into(),
                lat: 47.5,
                lon: -122.2,
                when_utc: Some("2026-07-17T18:35:03Z".into()),
                first_wall: Some("2026-07-17 11:35:03.000".into()),
                first_offset: "00:03.000".into(),
                visible_s: 2.5,
                frames: 5,
                best_conf: 97.0,
                mean_conf: 90.0,
                detector_score: Some(0.9),
                speed_mph: Some(31.0),
                bearing: Some(180.0),
                variants: vec![("ABC123".into(), 5)],
                image: Some(("x.jpg".into(), vec![0xFF, 0xD8, 0xFF, 0xD9])),
            }],
            unplaced: 1,
        };
        let doc = document(&[clip]);
        assert!(doc.contains("\"plate\":\"ABC123\""));
        assert!(doc.contains("[47.5,-122.2]"), "route in lat,lon order");
        assert!(doc.contains("data:image/jpeg;base64,/9j/2Q=="), "crop embedded");
        assert!(doc.contains("1 sighting(s) had no GPS fix"));
        assert!(doc.contains("unpkg.com/leaflet"));
    }

    #[test]
    fn script_closers_in_data_are_neutralised() {
        let clip = KmlClip {
            stem: "a</script><script>alert(1)".into(),
            camera: None,
            route: Vec::new(),
            placemarks: Vec::new(),
            unplaced: 0,
        };
        let doc = document(&[clip]);
        assert!(!doc.contains("a</script><script>alert(1)"));
    }
}
