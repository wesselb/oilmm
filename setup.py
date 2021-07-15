from setuptools import find_packages, setup

requirements = [
    "numpy>=1.16",
    "stheno>=1.1.3",
    "varz>=0.7.1",
    "backends>=1.3.6",
    "backends-matrix>=1.1.2",
    "plum-dispatch>=1.3.2",
]

setup(
    packages=find_packages(exclude=["docs"]),
    python_requires=">=3.6",
    install_requires=requirements,
    include_package_data=True,
)
