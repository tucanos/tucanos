use std::{env, path::PathBuf};

fn main() {
    // tucanos-ffi-test does not depends on the tucanos-ffi crate, only on libtucanos.so. This can
    // be declared using artifact-dependencies but this is still only nightly. Until it get stable
    // we manually trigger the tucanos-ffi build
    // See https://doc.rust-lang.org/nightly/cargo/reference/unstable.html#artifact-dependencies
    // See https://github.com/rust-lang/cargo/issues/6412
    let mut out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    for _ in 0..3 {
        out_path.pop();
    }
    println!("cargo:rustc-link-lib=tucanos");
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", out_path.display());
    println!("cargo:rustc-link-search=native={}", out_path.display());
    let header = out_path.join("tucanos.h");
    println!("cargo:rerun-if-changed={header:?}");
    bindgen::Builder::default()
        .header(header.to_str().unwrap())
        .generate()
        .expect("Unable to generate bindings")
        .write_to_file(out_path.join("bindgen.rs"))
        .expect("Couldn't write bindings!");
}
