//! Validates the ONNX detector/recognizer pair against a still image.
//!
//! cargo run --release --example alpr_test -- some_frame.png

use anyhow::Result;
use image::{imageops::FilterType, RgbImage};
use ort::session::Session;
use ort::value::Tensor;

const ALPHABET: &[u8] = b"0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_";
const DET_SIZE: u32 = 640;

/// Resize into a square canvas, preserving aspect, padding with YOLO's grey.
fn letterbox(img: &RgbImage, size: u32) -> (RgbImage, f32, u32, u32) {
    let scale = (size as f32 / img.width() as f32).min(size as f32 / img.height() as f32);
    let (nw, nh) = (
        (img.width() as f32 * scale).round() as u32,
        (img.height() as f32 * scale).round() as u32,
    );
    let resized = image::imageops::resize(img, nw, nh, FilterType::CatmullRom);
    let mut canvas = RgbImage::from_pixel(size, size, image::Rgb([114, 114, 114]));
    let (px, py) = ((size - nw) / 2, (size - nh) / 2);
    image::imageops::replace(&mut canvas, &resized, px as i64, py as i64);
    (canvas, scale, px, py)
}

fn main() -> Result<()> {
    ort::init_from("/opt/homebrew/lib/libonnxruntime.dylib")?.commit();
    let path = std::env::args().nth(1).expect("usage: alpr_test <image>");
    let img = image::open(&path)?.to_rgb8();
    println!("image {}x{}", img.width(), img.height());

    let mut det = Session::builder()?
        .commit_from_file("models/yolo-v9-t-640-license-plates-end2end.onnx")?;
    let mut rec = Session::builder()?.commit_from_file("models/cct_s_v2_global.onnx")?;

    let (canvas, scale, px, py) = letterbox(&img, DET_SIZE);
    let mut chw = vec![0f32; 3 * (DET_SIZE * DET_SIZE) as usize];
    let plane = (DET_SIZE * DET_SIZE) as usize;
    for (i, p) in canvas.pixels().enumerate() {
        chw[i] = p.0[0] as f32 / 255.0;
        chw[plane + i] = p.0[1] as f32 / 255.0;
        chw[2 * plane + i] = p.0[2] as f32 / 255.0;
    }
    let input = Tensor::from_array((vec![1i64, 3, DET_SIZE as i64, DET_SIZE as i64], chw))?;
    let outputs = det.run(ort::inputs!["images" => input])?;
    let (shape, data) = outputs["output0"].try_extract_tensor::<f32>()?;
    println!("detector output shape {shape:?}");

    let cols = 7;
    for row in data.chunks(cols) {
        println!("  raw row: {row:?}");
        let score = row[6];
        if score < 0.2 {
            continue;
        }
        // Layout: [batch, x1, y1, x2, y2, class, score]
        let unpad = |v: f32, pad: u32| ((v - pad as f32) / scale).max(0.0);
        let (x1, y1) = (unpad(row[1], px), unpad(row[2], py));
        let (x2, y2) = (unpad(row[3], px), unpad(row[4], py));
        let (w, h) = ((x2 - x1) as u32, (y2 - y1) as u32);
        if w == 0 || h == 0 {
            continue;
        }
        println!("  box ({x1:.0},{y1:.0}) {w}x{h} score {score:.3}");

        let crop = image::imageops::crop_imm(
            &img,
            (x1 as u32).min(img.width() - 1),
            (y1 as u32).min(img.height() - 1),
            w.min(img.width()),
            h.min(img.height()),
        )
        .to_image();
        let plate_img = image::imageops::resize(&crop, 128, 64, FilterType::Triangle);
        let bytes: Vec<u8> = plate_img.pixels().flat_map(|p| p.0).collect();
        let rin = Tensor::from_array((vec![1i64, 64, 128, 3], bytes))?;
        let rout = rec.run(ort::inputs!["input" => rin])?;
        let (pshape, pdata) = rout["plate"].try_extract_tensor::<f32>()?;
        println!("  plate head shape {pshape:?}, row0 sum {:.3}", pdata[..37].iter().sum::<f32>());

        let mut text = String::new();
        let mut confs = Vec::new();
        for slot in pdata.chunks(37).take(10) {
            let (idx, &p) = slot
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .expect("non-empty");
            let ch = ALPHABET[idx] as char;
            if ch != '_' {
                text.push(ch);
                confs.push(p);
            }
        }
        let mean = confs.iter().sum::<f32>() / confs.len().max(1) as f32;
        println!("  => PLATE '{text}' conf {mean:.3}");
    }
    Ok(())
}
