use std::{
    env,
    ffi::OsStr,
    path::{Path, PathBuf},
};

fn target_profile_dir(out_dir: &Path) -> PathBuf {
    out_dir
        .ancestors()
        .find(|path| path.file_name() == Some(OsStr::new("build")))
        .and_then(Path::parent)
        .expect("OUT_DIR does not contain a `build` ancestor directory")
        .to_path_buf()
}

fn main() {
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR is not set"));
    // tucanos-ffi-test does not depends on the tucanos-ffi crate, only on libtucanos.so. This can
    // be declared using artifact-dependencies but this is still only nightly. Until it get stable
    // we manually trigger the tucanos-ffi build
    // See https://doc.rust-lang.org/nightly/cargo/reference/unstable.html#artifact-dependencies
    // See https://github.com/rust-lang/cargo/issues/9096
    let artifact_dir = target_profile_dir(&out_dir);
    println!("cargo:rustc-link-lib=tucanos");
    #[cfg(any(target_os = "macos", target_os = "linux"))]
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", artifact_dir.display());
    println!("cargo:rustc-link-search=native={}", artifact_dir.display());
    let header = artifact_dir.join("tucanos.h");
    let header = header.to_str().expect("Invalid UTF-8 caracter in path");
    println!("cargo:rerun-if-changed={header}");
    bindgen::Builder::default()
        .header(header)
        .generate()
        .expect("Unable to generate bindings")
        .write_to_file(out_dir.join("bindgen.rs"))
        .expect("Couldn't write bindings!");
}
