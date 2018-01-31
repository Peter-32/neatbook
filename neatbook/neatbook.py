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
df = pd.read_csv("iris.csv") # Edit: Your dataset
print(df.describe(include = [np.number]))
print(df.describe(include = ['O']))
print(df.dtypes)
df.head()

"""

        header2 = """\
#### Initialize variables"""

        code2 = """\
from sklearn.model_selection import train_test_split
className = 'class' # Edit: Replace class with the Y column name
trainX, testX, trainY, testY = train_test_split(df.drop([className], axis=1),
                                                    df[className], train_size=0.75, test_size=0.25)

indexColumns = [] # Edit: Optionally add column names
skipColumns = [] # Edit: Optionally add column names

print("trainX\\n")
print(trainX.head())
print("\\ntrainY\\n")
print(trainY.head())
"""

        header3 = """\
#### Clean Data"""

        code3 = """\
from neatdata.neatdata import *

# Clean training set
neatdata =  NeatData()

cleanTrainX, cleanTrainY = neatdata.cleanTrainingDataset(trainX, trainY, indexColumns, skipColumns)

# Clean test set
cleanTestX = neatdata.cleanTestDataset(testX)

cleanTestY = neatdata.convertYToNumbersForModeling(testY)

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

tpot = TPOTClassifier(max_time_mins=5, # Edit: Set to 480 to train for 8 hours
                      population_size=100, max_eval_time_mins=5, verbosity=2)
tpot.fit(cleanTrainX, cleanTrainY)
print(tpot.score(cleanTestX, cleanTestY))
tpot.export('tpot_pipeline.py')

print("\\n\\nTPOT is done.")
"""

        header6 = """\
## Run this after TPOT is done

Creates the modelpipeline.py file.  That file also creates the trainedmodelpipeline.py.
"""

        code6 = """\
with open('modelpipeline.py', 'w') as fileOut:
    with open('tpot_pipeline.py', 'r') as fileIn:
        for line in fileIn:
            if line.startswith("import") or line.startswith("from "):
                fileOut.write(line)
    fileOut.write(\"\"\"from sklearn.metrics import accuracy_score
from neatdata.neatdata import *
from sklearn.metrics import confusion_matrix
import pickle

class ModelPipeline:

    def __init__(self):
        self.indexColumns, self.skipColumns = None, None
        self.neatData =  NeatData()
        self.className = 'class' # Edit: Replace class with the Y column name
        self.indexColumns = [] # Edit: Optionally add column names
        self.skipColumns = [] # Edit: Optionally add column names


    def execute(self):
        trainX, testX, trainY, testY = self._getDatasetFrom________() # Edit: choose one of two functions
        cleanTrainX, cleanTrainY, cleanTestX, cleanTestY = self._cleanDatasets()

    def _getDatasetFromOneFile(self):
        df = pd.read_csv('iris.csv') # Edit: Your dataset
        trainX, testX, trainY, testY = train_test_split(df.drop([self.className], axis=1),
                                                         df[self.className], train_size=0.75, test_size=0.25)
        return trainX, testX, trainY, testY

    def _getDatasetFromTwoFiles(self):
        trainingDf = pd.read_csv('train_iris.csv') # Edit: Your training dataset
        testDf = pd.read_csv('test_iris.csv') # Edit: Your test dataset
        trainX = trainingDf.drop([self.className], axis=1)
        trainY = trainingDf[self.className]
        testX = testDf.drop([self.className], axis=1)
        testY = testDf[self.className]
        return trainX, testX, trainY, testY

    def _cleanDatasets(self):
        cleanTrainX, cleanTrainY = self.neatData.cleanTrainingDataset(trainX, trainY, indexColumns, skipColumns)
        cleanTestX = self.neatData.cleanTestDataset(testX)
        cleanTestY = self.neatData.convertYToNumbersForModeling(testY)
        return cleanTrainX, cleanTrainY, cleanTestX, cleanTestY

    def _modelFit(self):
\"\"\")

showNextLines = False
with open('modelpipeline.py', 'a') as fileOut:
    with open('tpot_pipeline.py', 'r') as fileIn:
        for line in fileIn:
            if line.startswith("# Score"):
                showNextLines = True
            elif showNextLines and not line.startswith("exported_pipeline.fit") and not line.startswith("results"):
                fileOut.write("\\t\\t" + line)

with open('modelpipeline.py', 'a') as fileOut:
    fileOut.write(\"\"\"\\t\\texported_pipeline.fit(cleanTrainX, cleanTrainY)
\\t\\tresults = exported_pipeline.predict(cleanTestX)

    def printModelScores(self):
        print("Confusion Matrix:")
        print(confusion_matrix(cleanTestY, results))
        print(accuracy_score(cleanTestY, results))

    def createTrainedModelPipelineFile(self):

        def save_object(obj, filename):
            with open(filename, 'wb') as output:
                pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

        save_object(self, 'ModelPipeline.pkl')

        with open('trainedmodelpipeline.py', 'w') as fileOut:
            fileOut.write(\\\"\\\"\\\"

import pandas as pd
import pickle

class TrainedModelPipeline:

    def __init__(self):
        self.modelPipeline = None
        self.cleanTestX = None

    def execute(self):
        with open('ModelPipeline.pkl', 'rb') as input:
            self.modelPipeline = pickle.load(input)
        testX = self._getDataset()
        self.cleanTestX = self._cleanDataset(testX)
        results = self._predict()
        resultsDf = self._concatenatePredictionsToDataframe(results)
        self._saveResultsAsCSV(resultsDf)
        print("Done.  Created results.csv")

    def _getDataset(self):
        return pd.read_csv('test_iris.csv') # Edit: Your dataset

    def _cleanDataset(self, testX):
        return neatData.cleanTestDataset(testX)

    def _predict(self):
        results = exported_pipeline.predict(self.cleanTestX)
        return neatData.convertYToStringsOrNumbersForPresentation(results)

    def _concatenatePredictionsToDataframe(self, results):
        resultsDf = pd.DataFrame(results)
        return pd.concat([testX, resultsDf], axis=1)

    def _saveResultsAsCSV(self, resultsDf):
        resultsDf.to_csv('./results.csv')

trainedModelPipeline = new TrainedModelPipeline()
trainedModelPipeline.execute()        
\\\"\\\"\\\")

modelPipeline = new ModelPipeline()
modelPipeline.execute()


\"\"\")

print("Done creating modelpipeline.py")
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
