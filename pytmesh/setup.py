import setuptools
from setuptools_rust import Binding, RustExtension, build_rust


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
    name="pytmesh",
    version="0.1.0",
    packages=["pytmesh"],
    install_requires=["numpy"],
    extras_require={"test": ["vtk==9.3"]},
    rust_extensions=[
        RustExtension(
            "pytmesh.pytmesh",
            binding=Binding.PyO3,
        )
    ],
    cmdclass={"build_rust": BuildRustCommand},
    # rust extensions are not zip safe, just like C-extensions.
    zip_safe=False,
)
