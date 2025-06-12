import os
import sys
import subprocess
import glob
import shutil
from setuptools import setup, find_namespace_packages
from setuptools import Extension
from setuptools.command.build_ext import build_ext
from utils.get_torch_abi_flags import get_torch_abi_related_compiler_flags
import sysconfig
import torch

class CMakeExtension(Extension):
    def __init__(self, name, source_dir=".", cmake_args=None, **kwargs):
        super().__init__(name, sources=[])
        self.source_dir = os.path.abspath(source_dir)
        self.cmake_args = cmake_args or []

class CMakeBuild(build_ext):
    def build_extension(self, ext):
        # If it's not our CMake extension, use default behavior
        if not isinstance(ext, CMakeExtension):
            super().build_extension(ext)
            return

        # Prepare build directory
        build_dir = os.path.join(self.build_temp, ext.name)
        os.makedirs(build_dir, exist_ok=True)

        # Compute final extension output path and directory
        ext_path = self.get_ext_fullpath(ext.name)
        ext_dir = os.path.dirname(ext_path)

        # Configure CMake arguments to emit .so into ext_dir
        cmake_args = [
            f"-DCMAKE_BUILD_TYPE=Release",
            f"-DTORCH_INSTALL_PREFIX={sysconfig.get_paths()['purelib']}",
            f"-DCMAKE_PREFIX_PATH={torch.utils.cmake_prefix_path}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DENABLE_LOCAL_TT_METAL_BUILD=ON",
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={os.path.abspath(ext_dir)}",
            f"-DOUTPUT_NAME={ext.name.split('.')[-1]}",
            "-G", "Ninja",
        ]

        # Handle extra CMake flags from environment
        extra_cmake_flags = os.environ.get("CMAKE_FLAGS", "").split(";") if os.environ.get("CMAKE_FLAGS") else []
        if "-DENABLE_SUBMODULE_TT_METAL_BUILD=ON" in extra_cmake_flags:
            extra_cmake_flags.append("-DENABLE_LOCAL_TT_METAL_BUILD=OFF")
        torch_cxx_flags = get_torch_abi_related_compiler_flags()
        if torch_cxx_flags:
            flags_str = " ".join(torch_cxx_flags)
            extra_cmake_flags.append(f"-DCMAKE_CXX_FLAGS={flags_str}")
        cmake_args.extend(extra_cmake_flags)
        # User-specified CMake args
        cmake_args.extend(ext.cmake_args)

        # Run CMake configure and build
        subprocess.check_call(["cmake", ext.source_dir] + cmake_args, cwd=build_dir)
        subprocess.check_call(["cmake", "--build", ".", "--parallel"], cwd=build_dir)

        # Locate and copy the built .so into Python package directory
        libname = ext.name.split('.')[-1]
        built_so = os.path.join(ext_dir, f"{libname}.so")
        if not os.path.isfile(built_so):
            raise FileNotFoundError(f"Could not find built library at {built_so}")
        os.makedirs(ext_dir, exist_ok=True)
        shutil.copyfile(built_so, ext_path)
        print(f"Copied {built_so} → {ext_path}")

# Setup config
setup(
    name="ttnn_cpp_extension",
    version="0.1.0",
    packages=find_namespace_packages(include=["ttnn_cpp_extension*"]),
    ext_modules=[
        CMakeExtension(
            name="ttnn_cpp_extension.ttnn_device_extension",
            source_dir=".",
            cmake_args=[],
        ),
    ],
    cmdclass={"build_ext": CMakeBuild},
    python_requires=">=3.8",
)
