from setuptools import find_packages, setup

setup(
    name="sempy",
    version="0.1",
    description="SemPy - Spectral Element Methods in Python",
    license="BSD 3-clause",
    packages=find_packages(),
    package_data={"": ["meshes/*.msh"]},
    # Some older versions of these packages will likely work
    install_requires=[
        "scipy>=1.8", #_get_dtype() added in this version
        "numpy>=1.12.0", # Optimized einsums added in this version
        "meshio>=3.3.0",
        "matplotlib>=3.1.1",
        # 'mayavi>=4.6.0',
        "pytest>=5.2.2",
    ],
)
