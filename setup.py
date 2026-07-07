#!/usr/bin/env python

import os
import sys
import sysconfig
from fnmatch import fnmatch

from numpy import get_include as numpy_includes
from setuptools import Extension, setup

FREE_THREADED_PYTHON = sysconfig.get_config_var("Py_GIL_DISABLED") == 1


def c_sources(parent):
    sources = []
    for root, _, files in os.walk(parent):
        for f in files:
            fn = os.path.join(root, f)
            if fnmatch(fn, "*.c"):
                sources.append(fn)
    return sources


def c_includes(parent, depth=1):
    includes = [parent]
    for root, dirs, _ in os.walk(parent):
        for d in dirs:
            dn = os.path.join(root, d)
            if len(dn.split(os.sep)) - 1 > depth:
                continue
            includes.append(dn)
    return includes


SOURCES = c_sources("src")
INCLUDES = c_includes("include") + c_includes("src") + [numpy_includes()]

cfg = {
    "libraries": [],
    "define_macros": [],
    "extra_compile_args": [],
}
if not FREE_THREADED_PYTHON:
    cfg["define_macros"].append(
        ("Py_LIMITED_API", 0x03090000)  # PY_VERSION_HEX for 3.9
    )
    cfg["py_limited_api"] = True

if sys.platform == "win32":
    cfg["define_macros"].append(("__STDC__", 1))
    cfg["define_macros"].append(("_CRT_SECURE_NO_WARNINGS", None))
else:
    cfg["define_macros"].append(("NDEBUG", None))
    cfg["libraries"].append("m")
    cfg["extra_compile_args"] += [
        "-Wall",
        "-Wextra",
        "-Wpedantic",
        "-Wno-unused-parameter",
        "-Wincompatible-pointer-types",
    ]

# importing these extension modules is tested in `.github/workflows/build.yml`;
# when adding new modules here, make sure to add them to the `test_command` entry there
ext_modules = [
    Extension(
        "stsci.stimage._stimage",
        sources=SOURCES,
        include_dirs=INCLUDES,
        **cfg,
    ),
]

SETUPTOOLS_OPTIONS = {}
if not FREE_THREADED_PYTHON:
    SETUPTOOLS_OPTIONS["bdist_wheel"] = {"py_limited_api": "cp39"}

setup(
    ext_modules=ext_modules,
    options=SETUPTOOLS_OPTIONS,
)
