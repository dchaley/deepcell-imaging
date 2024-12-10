"""Installation script."""

import os

import setuptools


try:
    from Cython.Build import cythonize

    cy_ext = f"{os.extsep}pyx"
except ImportError:
    # this case is intended for use when installing from
    # a source distribution (produced with `sdist`),
    # which, as recommended by Cython documentation,
    # should include the generated `*.c` files,
    # in order to enable installation in absence of `cython`
    print("`import cython` failed")
    cy_ext = f"{os.extsep}c"


PACKAGE_NAME = "deepcell_imaging"
PACKAGE_SRC = "src/deepcell_imaging"


def run_setup():
    """Build and install package."""
    ext_modules = extensions()
    setuptools.setup(
        name=PACKAGE_NAME,
        ext_modules=ext_modules,
        # FIXME: use package discovery here
        packages=[
            PACKAGE_NAME,
            f"{PACKAGE_NAME}.gcp_batch_jobs",
            f"{PACKAGE_NAME}.image_processing",
            f"{PACKAGE_NAME}.utils",
        ],
        package_dir={PACKAGE_NAME: PACKAGE_SRC},
    )


def extensions():
    """Return C extensions, cythonize as needed."""
    import numpy

    extensions = dict(
        fast_hybrid=setuptools.extension.Extension(
            f"{PACKAGE_NAME}.image_processing.fast_hybrid_impl",
            sources=[f"{PACKAGE_SRC}/image_processing/fast_hybrid_impl{cy_ext}"],
        ),
        watershed_cy=setuptools.extension.Extension(
            f"{PACKAGE_NAME}.image_processing.watershed_cy",
            sources=[f"{PACKAGE_SRC}/image_processing/watershed_cy{cy_ext}"],
            extra_compile_args=["-DNPY_NO_DEPRECATED_API=NPY_1_9_API_VERSION"],
            include_dirs=[numpy.get_include()],
        ),
    )
    if cy_ext == f"{os.extsep}pyx":
        ext_modules = list()
        for k, v in extensions.items():
            c = cythonize(
                [v],
                # show_all_warnings=True,  # this line requires `cython >= 3.0`
            )
            ext_modules.append(c[0])
    else:
        ext_modules = list(extensions.values())
    return ext_modules


if __name__ == "__main__":
    run_setup()
