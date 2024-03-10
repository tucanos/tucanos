import setuptools
from setuptools_rust import Binding, RustExtension

features = []

setuptools.setup(
    name="pytucanos",
    version="0.0.1",
    packages=["pytucanos"],
    install_requires=["numpy", "matplotlib"],
    rust_extensions=[
        RustExtension(
            "pytucanos._pytucanos",
            binding=Binding.PyO3,
            features=features,
            debug=False,
        )
    ],
    # rust extensions are not zip safe, just like C-extensions.
    zip_safe=False,
)
