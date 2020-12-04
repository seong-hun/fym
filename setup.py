import os

from setuptools import find_packages, setup

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='fym',
    version='1.0.0',
    url='https://github.com/fdcl-nrf/fym',
    author='SNU FDCL',
    description='SNU FDCL Fym: Flight simulator for various purpose',
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='',
    include_package_data=True,
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        'gym',
        'matplotlib',
        'numdifftools',
        'h5py',
        'tqdm'
    ]
)
