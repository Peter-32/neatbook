import sys
import os
import nbformat as nbf
import re

class NeatNotebook:
    def __init__(self, ):
        PROJECT_FILE = os.path.realpath(os.path.basename(sys.argv[0]))
        PROJECT_PATH = re.match("(.*[/\\\])", PROJECT_FILE).group(1)
        PROJECT_NAME = re.match(".*[/\\\]+([^/\\\]+)[/\\\]+$", PROJECT_PATH).group(1)

        nb = nbf.v4.new_notebook()

        header1 = """\
# {}
#### Initialize variables""".format(PROJECT_NAME.capitalize())

        code1 = """\
from Neat.Neat import *


# Get the training set here
trainingSet =
# Get the test set here
testSet =
# Input the targetY value
targetY = '*** column_name_here ***'
# Optionally define these fields
indexColumns = []
skipColumns = []

trainingSet
"""

        header2 = """\
#### Clean Data"""

        code2 = """\
# Clean training set
neat =  Neat(self, trainingSet, targetY, indexColumns, skipColumns)
cleanTrainingSet = neat.df

# Clean test set
neat.cleanNewData(testSet)
cleanTestSet = neat.df
"""

        header3 = """\
#### Run TPOT"""

        code3 = """\
# Call TPOT

# Call script that takes the python code and loads everything into the next notebook section
"""

        header4 = """\
#### Reload the Page After TPOT Is Done"""

#IMPORTANT !@$!@#$!@#$!@#$
#        code4 = """\
## Copy/paste TPOT generated code here
#"""

        nb['cells'] = [nbf.v4.new_markdown_cell(header1),
                       nbf.v4.new_code_cell(code1),
                       nbf.v4.new_markdown_cell(header2),
                       nbf.v4.new_code_cell(code2),
                       nbf.v4.new_markdown_cell(header3),
                       nbf.v4.new_code_cell(code3),
                       nbf.v4.new_markdown_cell(header4) ]

        fname = '{}.ipynb'.format(PROJECT_PATH + PROJECT_NAME.capitalize() + "_Notebook")

        with open(fname, 'w') as f:
            nbf.write(nb, f)
