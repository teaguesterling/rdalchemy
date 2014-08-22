#!/usr/bin/env python

from setuptools import setup, find_packages

setup(name='rdalchemy',
      version='0.0.1',
      description='Using SQLAlchemy with chemical databases',
      packages=['rdalchemy'],
      install_requires=[
          'SQLAlchemy>=0.7.0',
      ],
)
