import os
import re

from setuptools import find_packages, setup

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

VERSIONFILE = "fym/_version.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

setup(
    name="fym",
    version=verstr,
    url="https://github.com/seong-hun/fym",
    author="Seong-hun Kim",
    author_email="kshoon92@gmail.com",
    description="Fym: An object-oriented simulator for dynamic systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    include_package_data=True,
    license_files=["LICENSE"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=["numpy", "scipy", "matplotlib", "numdifftools", "h5py", "tqdm"],
)
