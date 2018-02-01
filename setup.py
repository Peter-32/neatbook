from distutils.core import setup
setup(
  name = 'neatbook',
  packages = ['neatbook'], # this must be the same as the name above
  version = '0.14',
  description = 'One line of code that makes a notebook that writes code that writes code.',
  long_description='This automates nearly all the work for classification modeling. The notebook walks you through all the steps in a straightforward way. I wrote the code generation and data cleaning code. The last four steps of building a model are done by the TPOT package.',
  author = 'Peter Myers',
  author_email = 'peterjmyers1@gmail.com',
  url = 'https://github.com/Peter-32/neatbook', # use the URL to the github repo
  download_url = 'https://github.com/Peter-32/neatbook/archive/0.14.tar.gz', # I'll explain this in a second
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
