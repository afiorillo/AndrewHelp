from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

ext_modules = [
    Extension('FisheyeProc',
              ['FisheyeProc.pyx'],
              language="c++",  
              libraries=['../bin/x64/Release/FisheyeProc'],
              library_dirs=['.'])          
    ]

setup(
    name = 'FisheyeProc',
    cmdclass = {'build_ext': build_ext},
    ext_modules = ext_modules,
    include_dirs=[np.get_include()]
)