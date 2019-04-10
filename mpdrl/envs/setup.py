from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

import numpy

sourcefiles = ['mpdrl.pyx']

setup(
  cmdclass = {'build_ext' : build_ext},
  ext_modules = [Extension("mpdrl", sourcefiles)],
  include_dirs = [numpy.get_include()]
)
