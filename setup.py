#!/usr/bin/env python

#from distutils.core import setup
from setuptools import setup, find_packages, find_namespace_packages

setup(name='bozepy',
      version='1.0.14',
      description='Data reduction software for MSU astronomical data',
      author='David Nidever',
      author_email='dnidever@montana.edu',
      url='https://github.com/dnidever/bozepy',
      requires=['numpy','astropy','dlnpyutils','photutils'],
      zip_safe = False,
      include_package_data=True,
      packages=find_namespace_packages(where="python"),
      package_dir={"": "python"} 
)
