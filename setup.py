import os

from setuptools import find_packages, setup

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='fym',
    version='0.2.1',
    url='https://github.com/fdcl-nrf/fym',
    description='Flight simulator for various purpose',
    long_description=long_description,
    long_description_content_type='text/markdown',
    include_package_data=True,
    packages=find_packages(),
    install_requires=['gym', 'matplotlib', 'numdifftools']
)
