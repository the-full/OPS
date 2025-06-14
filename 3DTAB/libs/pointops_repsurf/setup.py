import os
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from distutils.sysconfig import get_config_vars

(opt,) = get_config_vars('OPT')
os.environ['OPT'] = " ".join(
    flag for flag in opt.split() if flag != '-Wstrict-prototypes'
)

setup(
    name='pointops_repsurf',
    version='1.0',
    packages=['pointops_repsurf'],
    package_dir={'pointops_repsurf': "functions"},
    ext_modules=[
        CUDAExtension(
            'pointops_repsurf_cuda', 
            [
                'src/pointops_api.cpp',

                'src/ballquery/ballquery_cuda.cpp',
                'src/ballquery/ballquery_cuda_kernel.cu',
                'src/knnquery/knnquery_cuda.cpp',
                'src/knnquery/knnquery_cuda_kernel.cu',
                'src/knnquery_heap/knnquery_heap_cuda.cpp',
                'src/knnquery_heap/knnquery_heap_cuda_kernel.cu',
                'src/grouping/grouping_cuda.cpp',
                'src/grouping/grouping_cuda_kernel.cu',
                'src/grouping_int/grouping_int_cuda.cpp',
                'src/grouping_int/grouping_int_cuda_kernel.cu',
                'src/interpolation/interpolation_cuda.cpp',
                'src/interpolation/interpolation_cuda_kernel.cu',
                'src/sampling/sampling_cuda.cpp',
                'src/sampling/sampling_cuda_kernel.cu',
            ],
            extra_compile_args={'cxx': ['-g'], 'nvcc': ['-O2']})
    ],
    cmdclass={'build_ext': BuildExtension})
