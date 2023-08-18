fn main() {
    let mut rpath = Vec::new();
    // Ensure transition of libOL1 RPATH to dependent crates
    // See https://github.com/jeromerobert/marechal-libol-sys#using
    if let Ok(ol_rpath) = std::env::var("DEP_OL_1_RPATH") {
        rpath.push(ol_rpath);
    }
    println!("cargo:rerun-if-env-changed=DEP_OL_1_RPATH");

    // See https://github.com/xgarnaud/libmeshb-sys#using
    if let Ok(meshb_rpath) = std::env::var("DEP_MESHB.7_RPATH") {
        rpath.push(meshb_rpath);
    }
    println!("cargo:rerun-if-env-changed=DEP_MESHB.7_RPATH");

    if let Ok(ld) = std::env::var("REMESH_LINK_DIRS") {
        for p in std::env::split_paths(&ld) {
            let s = p.display();
            rpath.push(s.to_string());
            println!("cargo:rustc-link-search={s}");
        }
    }
    println!("cargo:rerun-if-env-changed=REMESH_LINK_DIRS");

    if let Ok(lib) = std::env::var("REMESH_LIBRARIES") {
        for s in lib.split(',') {
            println!("cargo:rustc-link-lib={s}");
        }
    }
    println!("cargo:rerun-if-env-changed=REMESH_LIBRARIES");

    for p in &rpath {
        // Needed to build the tests
        #[cfg(any(target_os = "linux", target_os = "macos"))]
        println!("cargo:rustc-link-arg=-Wl,-rpath,{p}");
    }

    if !rpath.is_empty() {
        // non standard key
        // see https://doc.rust-lang.org/cargo/reference/build-script-examples.html#linking-to-system-libraries
        // and https://github.com/rust-lang/cargo/issues/5077
        println!("cargo:rpath={}", rpath.join(":"));
    }
}
