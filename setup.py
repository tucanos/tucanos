import setuptools
from setuptools_rust import Binding, RustExtension, build_rust
import sys


class BuildRustCommand(build_rust):
    user_options = build_rust.user_options + [
        ("features=", None, "Value for cargo --features")
    ]

    def initialize_options(self):
        super().initialize_options()
        self.features = None

    def finalize_options(self):
        super().finalize_options()
        ext = self.distribution.rust_extensions[0]
        ext.debug = self.debug
        ext.release = not self.debug
        if self.features:
            ext.features = self.features.split(",")


setuptools.setup(
    name="pytucanos",
    version="0.0.1",
    packages=["pytucanos"],
    install_requires=["numpy", "matplotlib"],
    rust_extensions=[
        RustExtension(
            "pytucanos._pytucanos",
            binding=Binding.PyO3,
        )
    ],
    cmdclass={"build_rust": BuildRustCommand},
    # rust extensions are not zip safe, just like C-extensions.
    zip_safe=False,
)
