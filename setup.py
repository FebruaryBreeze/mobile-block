"""
mobile-block
MobileNet-style blocks for PyTorch
Author: SF-Zhou
Date: 2019-01-15
"""

from setuptools import setup

name = 'mobile_block'
setup(
    name=name,
    version='0.0.6',
    description='MobileNet-style blocks for PyTorch',
    url=f'https://github.com/SF-Zhou/{name.replace("_", "-")}',
    author='SF-Zhou',
    author_email='sfzhou.scut@gmail.com',
    keywords='MobileNet Block PyTorch',
    py_modules=[f'{name}'],
    install_requires=['torch']
)
