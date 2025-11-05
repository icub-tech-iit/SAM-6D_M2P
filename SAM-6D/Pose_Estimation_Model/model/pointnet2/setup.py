# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import glob

_ext_src_root = "_ext_src"
_ext_sources = glob.glob("{}/src/*.cpp".format(_ext_src_root)) + glob.glob(
    "{}/src/*.cu".format(_ext_src_root)
)
_ext_headers = glob.glob("{}/include/*".format(_ext_src_root))

try:
    import torch
    import re
    current_list = os.environ.get("TORCH_CUDA_ARCH_LIST", "")
    current_cc = torch.cuda.get_device_capability(0)  # (major, minor)
    if torch.cuda.is_available() is False:
        raise RuntimeError("CUDA is not available")
    
    os.environ["TORCH_CUDA_ARCH_LIST"] = f"{current_cc[0]}.{current_cc[1]}"        
        
except Exception:
    pass  # fall back to whatever the environment provides

setup(
    name='pointnet2',
    packages = find_packages(),
    ext_modules=[
        CUDAExtension(
            name='pointnet2._ext',
            sources=_ext_sources,
            include_dirs = [os.path.join(_ext_src_root, "include")],
            extra_compile_args={
                "cxx": [],
                "nvcc": ["-O3", 
                "-DCUDA_HAS_FP16=1",
                "-D__CUDA_NO_HALF_OPERATORS__",
                "-D__CUDA_NO_HALF_CONVERSIONS__",
                "-D__CUDA_NO_HALF2_OPERATORS__",
            ]},)
    ],
    cmdclass={'build_ext': BuildExtension.with_options(use_ninja=False)}
)
