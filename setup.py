from distutils.core import setup
setup(
  name = 'neatbook',
  packages = ['neatbook'], # this must be the same as the name above
  version = '0.1',
  description = 'Run one command and a Jupyter notebook will appear with most of the work done for you in creating a classification machine learning model.  This library automates cleaning data.  Generates code that will clean data, alter features, and pick features, models, and parameters.'
  author = 'Peter Myers',
  author_email = 'peterjmyers1@gmail.com',
  url = 'https://github.com/peterldowns/mypackage', # use the URL to the github repo
  download_url = 'https://github.com/peterldowns/mypackage/archive/0.1.tar.gz', # I'll explain this in a second
  keywords = ['testing', 'logging', 'example'], # arbitrary keywords
  classifiers = [],
)
