from setuptools import setup, find_packages

__package_name__ = 'ATK'
__version__ = "0.1.0"


setup(
    name='ATK',
    version='0.1.0',
    author='Zhazi',
    author_email='zhazineedamail@163.com',
    description='ATK - ATK is a ToolKit designed for 3D deep learning adversarial research.',
    long_description=open('README.md', 'r', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    url='https://gitee.com/wronsky/ATK/tree/master',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'torch',
        'torchaudio',
        'torchvision',
        'tqdm',
        'vedo',
        'psutil',
        'scikit-learn',
        'h5py',
        'pyyaml',
        'einops',
        'trimesh',
        'ipdb',
        'setproctitle',
    ],
    extras_require={
        'numba': ['numba'],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: GNU/Linux :: Ubuntu',
        'Operating System :: GNU/Linux :: Deepin V23',
        'Programming Language :: Python :: 3.7',
        'Topic :: Utilities'
        'Natural Language :: Chinese (Simplified)'
    ],
    python_requires='>=3.7',  # 替换为你的包所要求的Python版本
)

