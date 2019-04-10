#!/usr/bin/env python

from setuptools import setup


install_requires = [
    'torch>=1.0.1',
    'Cython>=0.29.6',
    'gym>=0.12.1',
    'numpy>=1.16.2',
    'tqdm>=4.31.1'
]

setup(
    name='mpdrl',
    version='0.0.1',
    description='motion planning with deep reinforcement learning',
    url='https://github.com/Jumpei-Arima/mpdrl.git ',
    zip_safe=False,
    install_requires=install_requires,
)
