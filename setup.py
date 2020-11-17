from setuptools import setup, find_packages

setup(name='sempy',
      version='0.1',
      description='SemPy - Spectral Element Methods in Python',
      license='MIT',
      packages=find_packages(),
      package_data={'': ['meshes/*.msh']},
      install_requires=[
          'numpy==1.17.3',
          'meshio==3.3.0',
          'matplotlib==3.1.1',
          # 'mayavi==4.6.0',
          'pytest==5.2.2'
      ]
      )
