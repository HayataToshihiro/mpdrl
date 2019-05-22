#!/usr/bin/env python

from setuptools import setup


install_requires = [
    'torch==1.0.1',
    'Cython>=0.29.6',
    'gym==0.10.5',
    'numpy>=1.16.2',
    'tqdm>=4.32.1',
    'machina-rl>=0.2.0',
    'tensorboardX==1.6',
]

setup(
    name='mpdrl',
    version='0.0.1',
    description='motion planning with deep reinforcement learning',
    url='https://github.com/Jumpei-Arima/mpdrl.git ',
    zip_safe=False,
    install_requires=install_requires,
)
