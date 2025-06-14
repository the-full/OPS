from __future__ import division, absolute_import, with_statement, print_function
from setuptools import setup, find_packages
from distutils.extension import Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import glob
import os.path as osp

try:
    import builtins
except:
    import __builtin__ as builtins

builtins.__POINTNET2_SETUP__ = True
import utils

_cur_dir = osp.dirname(osp.abspath(__file__))
_ext_src_root = osp.join(_cur_dir, "utils/_ext-src")
_ext_sources = glob.glob("{}/src/*.cpp".format(_ext_src_root)) + glob.glob(
    "{}/src/*.cu".format(_ext_src_root)
)
_ext_headers = glob.glob("{}/include/*".format(_ext_src_root))

requirements = ["h5py", "enum34", "future"]

setup(
    name="pointops_rscnn",
    version="2.1.1",
    author="Erik Wijmans",
    packages=[
        'pointops_rscnn', 
        'pointops_rscnn.pytorch_utils'
    ],
    package_dir={
        'pointops_rscnn': 'utils', 
        'pointops_rscnn.pytorch_utils': 'utils/pytorch_utils'
    },
    install_requires=requirements,
    ext_modules=[
        CUDAExtension(
            name="_ext",
            sources=_ext_sources,
            extra_compile_args={
                "cxx": ["-O2", "-I{}".format("{}/include".format(_ext_src_root))],
                "nvcc": ["-O2", "-I{}".format("{}/include".format(_ext_src_root))],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
