from setuptools import find_packages, setup

REQUIRED_PACKAGES = []

setup(
    name="pkg",
    packages=find_packages(),
    version="0.1.0",
    description="Local package for mouse paper",
    author="Neurodata",
    license="MIT",
    install_requires=REQUIRED_PACKAGES,
    dependency_links=[],
)
