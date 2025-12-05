fn main() {
    #[cfg(any(target_os = "linux", target_os = "macos"))]
    if let Ok(rpath) = std::env::var("DEP_TUCANOS_RPATH") {
        for s in rpath.split(':') {
            println!("cargo:rustc-link-arg=-Wl,-rpath,{s}");
        }
    }
}
