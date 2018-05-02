from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy
import scipy

extensions = [
    Extension("predator_prey__cython",["predator_prey_numpy.pyx"],
    include_dirs=[numpy.get_include(),"."]),

]

setup(
    name='predator_prey',
    version='0.0.1',
    description='''Cython support for Predator Prey Env''',
    url='...',
    author='',
    author_email='',
    install_requires=[
        'cython',
        'numpy',
        'scipy',
    ],
    ext_modules=cythonize(extensions),
)