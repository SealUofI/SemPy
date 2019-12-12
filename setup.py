from setuptools import setup,find_packages

setup(name='sempy',
    version='0.1',
    description='SemPy - Spectral Element Methods in Python',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        'numpy==1.17.3',
        'meshio==3.3.0'
    ]
)
