from setuptools import setup

setup(
    name='fym',
    version='0.2.3',
    description='SNU FDCL Fym: A gym for flight dynamics',
    author='SNU FDCL',
    url='https://github.com/fdcl-nrf/fym',
    license='',
    install_requires=['gym', 'matplotlib', 'numdifftools'],
    packages=['fym']
)
