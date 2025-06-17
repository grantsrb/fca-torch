from setuptools import setup, find_packages

setup(name='fca',
      packages=find_packages(),
      version="0.1.0",
      description='Functional components analysis tools',
      author='Satchel Grant',
      author_email='grantsrb@stanford.edu',
      url='https://github.com/grantsrb/fca.git',
      install_requires= ["numpy", "torch", ],
      py_modules=['fca'],
      long_description='''
          This package contains code to easily setup a functional components analysis object for finding functionally sufficient bottleneck representations for any given layer of a module. 
          ''',
      classifiers=[
          'Intended Audience :: Science/Research',
          'Operating System :: MacOS :: MacOS X :: Ubuntu',
          'Topic :: Scientific/Engineering :: Information Analysis'],
      )
