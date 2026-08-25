"""Installation script for the 'physics_ot_tasks' python package."""

import os

import toml
from setuptools import setup

EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))
EXTENSION_TOML_DATA = toml.load(os.path.join(EXTENSION_PATH, "config", "extension.toml"))

INSTALL_REQUIRES = [
    "physics-ot",
    "pyyaml",
]

setup(
    name="physics_ot_tasks",
    packages=["physics_ot_tasks"],
    author=EXTENSION_TOML_DATA["package"]["author"],
    maintainer=EXTENSION_TOML_DATA["package"]["maintainer"],
    version=EXTENSION_TOML_DATA["package"]["version"],
    description=EXTENSION_TOML_DATA["package"]["description"],
    keywords=EXTENSION_TOML_DATA["package"]["keywords"],
    install_requires=INSTALL_REQUIRES,
    license="BSD-3-Clause",
    include_package_data=True,
    python_requires=">=3.10",
    zip_safe=False,
)
