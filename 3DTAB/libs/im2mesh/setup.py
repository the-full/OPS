from setuptools import setup, find_packages
from distutils.extension import Extension
from Cython.Build import cythonize
from torch.utils.cpp_extension import BuildExtension
import numpy


# Get the numpy include directory.
numpy_include_dir = numpy.get_include()

# Extensions
# mcubes (marching cubes algorithm)
mcubes_module = Extension(
    'im2mesh.libmcubes.mcubes',
    sources=[
        'im2mesh/libmcubes/mcubes.pyx',
        'im2mesh/libmcubes/pywrapper.cpp',
        'im2mesh/libmcubes/marchingcubes.cpp'
    ],
    language='c++',
    extra_compile_args=['-std=c++11'],
    include_dirs=[numpy_include_dir]
)

# triangle hash (efficient mesh intersection)
triangle_hash_module = Extension(
    'im2mesh.libmesh.triangle_hash',
    sources=[
        'im2mesh/libmesh/triangle_hash.pyx'
    ],
    libraries=['m']  # Unix-like specific
)

# mise (efficient mesh extraction)
mise_module = Extension(
    'im2mesh.libmise.mise',
    sources=[
        'im2mesh/libmise/mise.pyx'
    ],
    include_dirs=[numpy_include_dir]
)

# simplify (efficient mesh simplification)
simplify_mesh_module = Extension(
    'im2mesh.libsimplify.simplify_mesh',
    sources=[
        'im2mesh/libsimplify/simplify_mesh.pyx'
    ],
    include_dirs=[numpy_include_dir]
)

# Gather all extension modules
ext_modules = [
    mcubes_module,
    mise_module,
    simplify_mesh_module,
    triangle_hash_module,
]

setup(
    name='im2mesh',
    version='0.1',
    packages=find_packages(),
    ext_modules=cythonize(ext_modules),
    cmdclass={
        'build_ext': BuildExtension
    },
    include_dirs=[numpy_include_dir]
)
