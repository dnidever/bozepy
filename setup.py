#!/usr/bin/env python

#from distutils.core import setup
from setuptools import setup, find_packages

setup(name='bozepy',
      version='1.0',
      description='Data reduction software for MSU's Celestron+SBIG astronomical data',
      author='David Nidever',
      author_email='dnidever@montana.edu',
      url='https://github.com/dnidever/bozepy',
      packages=find_packages(exclude=["tests"]),
      requires=['numpy','astropy','dlnpyutils'],
#      include_package_data=True,
)
