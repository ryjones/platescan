//! Prints the input/output signature of an ONNX model.
//!
//! cargo run --release --example probe -- models/*.onnx

use ort::session::Session;

fn main() -> anyhow::Result<()> {
    ort::init_from("/opt/homebrew/lib/libonnxruntime.dylib")?.commit();
    for path in std::env::args().skip(1) {
        let session = Session::builder()?.commit_from_file(&path)?;
        println!("== {path}");
        for i in session.inputs() {
            println!("  in   {:<20} {:?}", i.name(), i.dtype());
        }
        for o in session.outputs() {
            println!("  out  {:<20} {:?}", o.name(), o.dtype());
        }
    }
    Ok(())
}
