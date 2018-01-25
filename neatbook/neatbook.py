import sys
import os
import nbformat as nbf
import re

class Neatbook:
    def __init__(self, ):
        PROJECT_FILE = os.path.realpath(os.path.basename(sys.argv[0]))
        PROJECT_PATH = re.match("(.*[/\\\])", PROJECT_FILE).group(1)
        PROJECT_NAME = re.match(".*[/\\\]+([^/\\\]+)[/\\\]+$", PROJECT_PATH).group(1)

        nb = nbf.v4.new_notebook()

        header1 = """\
# {} Neatbook
#### Get Data""".format(PROJECT_NAME.capitalize())

        code1 = """\
import pandas as pd
import numpy as np

# Get data here
df = pd.read_csv("iris.csv") ## Edit: Your dataset
print(df.describe(include = [np.number]))
print(df.describe(include = ['O']))
print(df.dtypes)
df.head()

"""

        header2 = """\
#### Initialize variables"""

        code2 = """\
from sklearn.model_selection import train_test_split
className = 'class' ## Edit: Replace class with the Y column name
trainX, testX, trainY, testY = train_test_split(df.drop([className], axis=1),
                                                    df[className], train_size=0.75, test_size=0.25)

indexColumns = [] ## Edit: Optionally add column names
skipColumns = [] ## Edit: Optionally add column names

print("trainX\\n")
print(trainX.head())
print("\\ntrainY\\n")
print(trainY.head())
"""

        header3 = """\
#### Clean Data"""

        code3 = """\
from neatbook.neat import *

# Clean training set
neat =  Neat(trainX, trainY, indexColumns, skipColumns)
cleanTrainX = neat.df
cleanTrainY = neat.trainY

# Clean test set
neat.cleanNewData(testX)
cleanTestX = neat.df
cleanTestY = neat.getYAsNumber(testY)

print("Cleaning done")
"""

        header4 = """\
#### Review Cleaned Data"""

        code4 = """\
print(cleanTrainX.describe(include = [np.number]))
print(cleanTrainX.head())

print(cleanTrainY)


print(cleanTestX.describe(include = [np.number]))
print(cleanTestX.head())


print(cleanTestY)
"""

        header5 = """\
#### Run TPOT"""

        code5 = """\
from tpot import TPOTClassifier

tpot = TPOTClassifier(max_time_mins=5, ## Edit: Set to 480 to train for 8 hours
                      population_size=100, max_eval_time_mins=5, verbosity=2)
tpot.fit(cleanTrainX, cleanTrainY)
print(tpot.score(cleanTestX, cleanTestY))
tpot.export('tpot_pipeline.py')

print("\\n\\nTPOT is done.")
"""

        header6 = """\
## Run this after TPOT is done

Creates the Python_Training_Test.py file.  That file creates the optional Python_Test.py file.

- **Python_Training_Test.py:** Train the model from TPOT.  Test it on a test set.
- **Python_Test.py:** Used to test new data without model training.  The model is saved to disk during the Python_Training_Test.py script run."""

        code6 = """\
with open('Python_Training_Test.py', 'w') as fileOut:
    with open('tpot_pipeline.py', 'r') as fileIn:
        for line in fileIn:
            if line.startswith("import") or line.startswith("from "):
                fileOut.write(line)
    fileOut.write(\"\"\"from sklearn.metrics import accuracy_score
from neatbook.neat import *
from sklearn.metrics import confusion_matrix
import pickle


##### IF YOU HAVE 1 DATASET UNCOMMENT THIS CODE: #####

# df = pd.read_csv('iris.csv') ## Edit: Your dataset
# className = 'class' ## Edit: Replace class with the Y column name
# trainX, testX, trainY, testY = train_test_split(df.drop([className], axis=1),
#                                                     df[className], train_size=0.75, test_size=0.25)

#######################################################

##### IF YOU HAVE 2 DATASETS UNCOMMENT THIS CODE: #####

# trainDf = pd.read_csv('train_iris.csv') ## Edit: Your dataset
# testDf = pd.read_csv('test_iris.csv') ## Edit: Your dataset

# className = 'class' ## Edit: Replace class with the Y column name
# trainX = trainDf.drop([className], axis=1)
# trainY = trainDf[className]
# testX = testDf.drop([className], axis=1)
# testY = testDf[className]

#######################################################

################### Set Variables: ####################

indexColumns = [] ## Edit: Optionally add column names
skipColumns = [] ## Edit: Optionally add column names

#######################################################

####################### Clean: ########################

# Clean training set
neat =  Neat(trainX, trainY, indexColumns, skipColumns)
cleanTrainX = neat.df
cleanTrainY = neat.getYAsNumber(trainY)

# Clean test set
neat.cleanNewData(testX)
cleanTestX = neat.df
cleanTestY = neat.getYAsNumber(testY)

#######################################################

###################### Pipeline: ######################

\"\"\")

showNextLines = False
with open('Python_Training_Test.py', 'a') as fileOut:
    with open('tpot_pipeline.py', 'r') as fileIn:
        for line in fileIn:
            if line.startswith("# Score"):
                showNextLines = True
            elif showNextLines and not line.startswith("exported_pipeline.fit") and not line.startswith("results"):
                fileOut.write(line)

with open('Python_Training_Test.py', 'a') as fileOut:
    fileOut.write(\"\"\"exported_pipeline.fit(cleanTrainX, cleanTrainY)
results = exported_pipeline.predict(cleanTestX)

#######################################################

################## Confusion Matrix: ##################

print("Confusion Matrix:")
print(confusion_matrix(cleanTestY, results))
print(accuracy_score(cleanTestY, results))

#######################################################

############ Create Python_Test.py File: ##############

def save_object(obj, filename):
    with open(filename, 'wb') as output:
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

save_object(neat, 'neat.pkl')
save_object(exported_pipeline, 'exported_pipeline.pkl')
save_object(indexColumns, 'indexColumns.pkl')
save_object(skipColumns, 'skipColumns.pkl')
save_object(className, 'className.pkl')


with open('Python_Test.py', 'w') as fileOut:
    fileOut.write(\\\"\\\"\\\"

import pandas as pd
import pickle

#################### Get Dataset: #####################

testX = pd.read_csv('test_iris.csv') ## Edit: Your dataset

#######################################################

################### Set Variables: ####################

with open('neat.pkl', 'rb') as input:
    neat = pickle.load(input)
with open('exported_pipeline.pkl', 'rb') as input:
    exported_pipeline = pickle.load(input)
with open('indexColumns.pkl', 'rb') as input:
    indexColumns = pickle.load(input)
with open('skipColumns.pkl', 'rb') as input:
    skipColumns = pickle.load(input)
with open('className.pkl', 'rb') as input:
    className = pickle.load(input)

#######################################################

####################### Clean: ########################

neat.cleanNewData(testX)
cleanTestX = neat.df

#######################################################

###################### Predict: #######################

results = exported_pipeline.predict(cleanTestX)
resultsDf = pd.DataFrame(results)
submitDf = pd.concat([testX, resultsDf], axis=1)
submitDf.to_csv('./submit.csv')
print("Done")
print(results)

#######################################################

#######################################################
\\\"\\\"\\\")
\"\"\")

print("Done creating your Python_Training_Test.py")
"""

        nb['cells'] = [nbf.v4.new_markdown_cell(header1),
                       nbf.v4.new_code_cell(code1),
                       nbf.v4.new_markdown_cell(header2),
                       nbf.v4.new_code_cell(code2),
                       nbf.v4.new_markdown_cell(header3),
                       nbf.v4.new_code_cell(code3),
                       nbf.v4.new_markdown_cell(header4),
                       nbf.v4.new_code_cell(code4),
                       nbf.v4.new_markdown_cell(header5),
                       nbf.v4.new_code_cell(code5),
                       nbf.v4.new_markdown_cell(header6),
                       nbf.v4.new_code_cell(code6) ]

        fname = '{}.ipynb'.format(PROJECT_PATH + PROJECT_NAME.capitalize() + "_Neatbook")

        if not os.path.isfile(fname):
            with open(fname, 'w') as f:
                nbf.write(nb, f)
