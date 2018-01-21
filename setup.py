from distutils.core import setup
setup(
  name = 'neatbook',
  packages = ['neatbook'], # this must be the same as the name above
  version = '0.3',
  description = 'Automates nearly all the work for classification modeling with help from other packages',
  long_description='This package creates a Python notebook that automates nearly all the work for classification modeling with help from other packages. I wrote the data cleaning code and notebook generation code. The last four steps of building a model are done by the TPOT package.  It can save me hours and makes the work straightforward.',
  author = 'Peter Myers',
  author_email = 'peterjmyers1@gmail.com',
  url = 'https://github.com/Peter-32/neatbook', # use the URL to the github repo
  download_url = 'https://github.com/Peter-32/neatbook/archive/0.3.tar.gz', # I'll explain this in a second
  keywords = ['automated machine learning', 'cleaning', 'classification', 'code generation'],
  classifiers = [
  'Development Status :: 3 - Alpha',
'Topic :: Scientific/Engineering :: Artificial Intelligence',
'License :: OSI Approved :: MIT License',
'Programming Language :: Python :: 3',
'Programming Language :: Python :: 3.2',
'Programming Language :: Python :: 3.3',
'Programming Language :: Python :: 3.4',
'Programming Language :: Python :: 3.5',
'Programming Language :: Python :: 3.6',
  ],
  install_requires=['nbformat', 'sklearn', 'numpy', 'pandas'],
  python_requires='>=3',
  license='MIT'
)
