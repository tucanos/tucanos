# adapted from https://github.com/python-pillow/Pillow/blob/main/_custom_build/backend.py
# see also https://peps.python.org/pep-0517/#in-tree-build-backends
import sys
from setuptools.build_meta import build_wheel, build_editable

FEATURES = ["libmeshb", "nlopt", "metis", "scotch"]


def update_argv(config_settings):
    if config_settings:
        flags = []
        if config_settings.get("debug", "false").lower() == "true":
            flags += ["--debug"]
        else:
            flags += ["--release"]
        features = []
        for feature in FEATURES:
            if config_settings.get(feature, "false").lower() == "true":
                features.append(feature)
        if len(features) > 0:
            flags += ["--features=%s" % ",".join(features)]
        if flags:
            sys.argv = sys.argv[:1] + ["build_rust"] + flags + sys.argv[1:]


backend_class = build_wheel.__self__.__class__


class _CustomBuildMetaBackend(backend_class):
    def run_setup(self, setup_script="setup.py"):
        update_argv(self.config_settings)
        return super().run_setup(setup_script)

    def build_wheel(
        self, wheel_directory, config_settings=None, metadata_directory=None
    ):
        self.config_settings = config_settings
        return super().build_wheel(wheel_directory, config_settings, metadata_directory)


build_wheel = _CustomBuildMetaBackend().build_wheel

backend_class = build_editable.__self__.__class__


class _CustomBuildMetaBackend(backend_class):
    def run_setup(self, setup_script="setup.py"):
        update_argv(self.config_settings)
        return super().run_setup(setup_script)

    def build_editable(
        self, wheel_directory, config_settings=None, metadata_directory=None
    ):
        self.config_settings = config_settings
        return super().build_editable(
            wheel_directory, config_settings, metadata_directory
        )


build_editable = _CustomBuildMetaBackend().build_editable
