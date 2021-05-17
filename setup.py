import os

from setuptools import find_packages, setup

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='fym',
    version='1.1.3',
    url='https://github.com/fdcl-nrf/fym',
    author='SNU FDCL',
    author_email='kshoon92@gmail.com',
    description='SNU FDCL Fym: Flight simulator for various purpose',
    long_description=long_description,
    long_description_content_type='text/markdown',
    include_package_data=True,
    license_files=['LICENSE'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'numdifftools',
        'h5py',
        'tqdm'
    ]
)
