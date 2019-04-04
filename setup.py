#!/usr/bin/env python

from setuptools import setup


install_requires = [
    'torch>=1.0.1',
    'gym>=0.10.9',
    'numpy>=1.13.3'
]

setup(
    name='mpdrl',
    version='0.0.1',
    description='motion planning with deep reinforcement learning',
    url='https://github.com/Jumpei-Arima/mpdrl.git ',
    zip_safe=False,
    install_requires=install_requires,
)
