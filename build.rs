fn main() {
    if let Ok(rpath) = std::env::var("DEP_TUCANOS_RPATH") {
        for s in rpath.split(':') {
            #[cfg(any(target_os = "linux", target_os = "macos"))]
            println!("cargo:rustc-link-arg=-Wl,-rpath,{s}");
        }
    }
}
