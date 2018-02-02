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
df = pd.read_csv("train.csv") # Edit: Your dataset
# classDF = pd.read_csv("train_labels.csv", header=None)
# df = pd.concat([df, classDF], axis=1)
print(df.shape)
print(df.describe(include = [np.number]))
print(df.dtypes)
print(df.describe(include = ['O']))
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
iWillManuallyCleanColumns = [] # Edit: Optionally add column names

print("trainX.shape = ", trainX.shape)
print("testX.shape = ", testX.shape)
print("trainY.shape = ", trainY.shape)
print("testY.shape = ", testY.shape)
print("\ntrainY\n")
print(trainY.head())
print("trainX\n")
print(trainX.head())
"""

        header3 = """\
#### Clean Data"""

        code3 = """\
from neatdata.neatdata import *

neatdata =  NeatData()
cleanTrainX, cleanTrainY = neatdata.cleanTrainingDataset(trainX, trainY, indexColumns, iWillManuallyCleanColumns)
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
        self.indexColumns, self.iWillManuallyCleanColumns = None, None
        self.neatData =  NeatData()
        self.className = 'class' # Edit: Replace class with the Y column name
        self.indexColumns = [] # Edit: Optionally add column names
        self.iWillManuallyCleanColumns = [] # Edit: Optionally add column names
        self.cleanTrainX, self.cleanTrainY, self.cleanTestX, self.cleanTestY = None, None, None, None
        self.results = None


    def execute(self):
        trainX, testX, trainY, testY = self._getDatasetFrom________() # Edit: choose one of two functions
        self._cleanDatasets(trainX, testX, trainY, testY)
        self._modelFit()
        self._printModelScores()
        self._createTrainedModelPipelineFile()
        self._saveObjectsToDisk()
        self._createTrainedModelPipelineFile()

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

    def _cleanDatasets(self, trainX, testX, trainY, testY):
        self.cleanTrainX, self.cleanTrainY = self.neatData.cleanTrainingDataset(trainX, trainY, self.indexColumns, self.iWillManuallyCleanColumns)
        self.cleanTestX = self.neatData.cleanTestDataset(testX)
        self.cleanTestY = self.neatData.convertYToNumbersForModeling(testY)

    def _modelFit(self):
\"\"\")

showNextLines = False
with open('modelpipeline.py', 'a') as fileOut:
    with open('tpot_pipeline.py', 'r') as fileIn:
        for line in fileIn:
            if line.startswith("# Score"):
                showNextLines = True
            elif showNextLines and not line.startswith("exported_pipeline.fit") and not line.startswith("results"):
                fileOut.write("        " + line)

with open('modelpipeline.py', 'a') as fileOut:
    fileOut.write(\"\"\"        self.exported_pipeline = exported_pipeline
        self.exported_pipeline.fit(self.cleanTrainX, self.cleanTrainY)
        self.results = self.exported_pipeline.predict(self.cleanTestX)

    def _printModelScores(self):
        print("Confusion Matrix:")
        print(confusion_matrix(self.cleanTestY, self.results))
        print(accuracy_score(self.cleanTestY, self.results))

    def _saveObjectsToDisk(self):
        def save_object(obj, filename):
            with open(filename, 'wb') as output:
                pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

        save_object(self.exported_pipeline, 'exportedPipeline.pkl')
        save_object(self.neatData, 'NeatData.pkl')

    def _createTrainedModelPipelineFile(self):
        with open('trainedmodelpipeline.py', 'w') as fileOut:
            fileOut.write(\\\"\\\"\\\"

import pandas as pd
import pickle

class TrainedModelPipeline:

    def __init__(self):
        self.exportedPipeline = None
        self.neatData = None
        self.testX = None
        self.cleanTestX = None
        self.results = None
        self.resultsDf = None

    def execute(self):
        self._loadObjects()
        self._getDataset()
        self._cleanDataset()
        self._predict()
        self._concatenatePredictionsToDataframe()
        self._saveResultsAsCSV()
        print("Done. Created results.csv")

    def _loadObjects(self):
        with open('exportedPipeline.pkl', 'rb') as input:
            self.exportedPipeline = pickle.load(input)
        with open('NeatData.pkl', 'rb') as input:
            self.neatData = pickle.load(input)

    def _getDataset(self):
        self.testX = pd.read_csv('test_iris.csv') # Edit: Your dataset

    def _cleanDataset(self):
        self.cleanTestX = self.neatData.cleanTestDataset(self.testX)

    def _predict(self):
        self.results = self.exportedPipeline.predict(self.cleanTestX)
        self.results = self.neatData.convertYToStringsOrNumbersForPresentation(self.results)

    def _concatenatePredictionsToDataframe(self):
        self.resultsDf = pd.DataFrame(self.results)
        self.resultsDf = pd.concat([self.testX, self.resultsDf], axis=1)

    def _saveResultsAsCSV(self):
        self.resultsDf.to_csv('./results.csv')

trainedModelPipeline = TrainedModelPipeline()
trainedModelPipeline.execute()
\\\"\\\"\\\")

modelPipeline = ModelPipeline()
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
