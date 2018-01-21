import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.utils import resample
from math import ceil

class Neat:

    def __init__(self, trainX, trainY, indexColumns=[], skipColumns=[]):
        if len(trainY) != len(trainX.index):
            raise Exception('Error: trainX and trainY are differing lengths')
            return
        self.df = trainX
        self.trainY = np.array( trainY )
        self.indexColumns = self._cleanColumnNamesArray(indexColumns)
        self.skipColumns = self._cleanColumnNamesArray(skipColumns)
        self.newData = None
        self.trainYMappingsStrToNum = {'NotFound': -99}
        self.trainYMappingsNumToStr = {-99: 'NotFound'}
        self.numberColumns = []
        self.categoryColumns = []
        self.datetimeColumns = []
        self.medians = []
        self.lowerBounds = []
        self.upperBounds = []
        self.uniqueCategoryValues = {}
        self.valuesThatDontMapTo_Other = {}
        self.categoryFrequencies = {}
        self.trainYFrequencies = {}
        self.trainYUpsamplesNeeded = {}
        self.finalColumnNames = []
        self.columnsDropped = []
        # TrainY
        self._setTrainYMappings()
        self._convertTrainYToNumeric()
        self._dropNATrainYRows()
        # Column Metadata
        self._cleanColumnNamesDF()
        self._setColumnDataTypes()
        # Index
        self._dropDuplicatesAndMissingRowsIfIndexIsSpecified()
        # Datetimes
        self._convertDatetimeToNumber()
        # Numbers
        self._saveMediansAndBounds()
        self._fixMissingNumValuesAndInfinity()
        self._fixHighLeveragePoints()
        # Categories
        self._saveUniqueCategoryValues()
        self._saveCategoryFrequenciesAndValuesThatDontMapTo_Other()
        self._dropCategoryColumnsWithAllMissingValues()
        self._fixMissingCategoryValuesAndMapValuesTo_Other()
        self._applyOneHotEncoding()
        # Class Imbalance
        self._saveTrainYFrequencies()
        self._saveTrainYUpsamplesNeeded()
        self._fixTrainYImbalance()
        # Index
        self._addIndex()
        # Get Final Column Names
        self._saveFinalColumnNames()

    def cleanNewData(self, newData):
        self.df = newData
        # Column Metadata
        self._cleanColumnNamesDF()
        # Datetimes
        self._convertDatetimeToNumber()
        # Numbers
        self._fixMissingNumValuesAndInfinity()
        self._fixHighLeveragePoints()
        # Categories
        self._fixMissingCategoryValuesAndMapValuesTo_Other()
        self._applyOneHotEncoding()
        # Index
        self._addIndex()
        # New Data
        self._newDataDropDroppedColumns()
        self._newDataAddMissingFinalColumnNames()
        self._newDataDropExtraColumnNames()

    def getYAsString(self, newDataY):
        output = None
        newDataYAsNumpy = np.array( newDataY )
        if newDataYAsNumpy.dtype.kind in {'U', 'S'}: # a string
            for i in range(0,len(newDataY)):
                if newDataY[i] not in trainYMappingsNumToStr.values():
                    newDataY[i] = 'NotFound'
            newDataYAsNumpy = np.array( newDataY )
            output = newDataYAsNumpy

        else:
            for i in range(0,len(newDataY)):
                if newDataY[i] not in trainYMappingsNumToStr.keys():
                    newDataY[i] = -99
            newDataYAsNumpy = np.array( newDataY )
            output = np.vectorize(self.trainYMappingsNumToStr.get)(newDataYAsNumpy)
        return output

    def getYAsNumber(self, newDataY):
        output = None
        newDataYAsNumpy = np.array( newDataY )
        if newDataYAsNumpy.dtype.kind in {'U', 'S'}: # a string
            for i in range(0,len(newDataY)):
                if newDataY[i] not in trainYMappingsStrToNum.keys():
                    newDataY[i] = 'NotFound'
            newDataYAsNumpy = np.array( newDataY )
            output = np.vectorize(self.trainYMappingsStrToNum.get)(newDataYAsNumpy)

        else:
            for i in range(0,len(newDataY)):
                if newDataY[i] not in trainYMappingsStrToNum.values():
                    newDataY[i] = -99
            newDataYAsNumpy = np.array( newDataY )
            output = newDataYAsNumpy
        return output

    ########## Getting Started ##########

    def _cleanColumnNamesArray(self, columns):
        if type(columns) == str:
            columns = [columns]
        arr = []
        for column in columns:
            arr.append(self._cleanColumnName(column))
        return arr

    def _cleanColumnName(self, string):
        return string.strip().lower().replace(' ', '_')

    ########## TrainY ##########

    def _setTrainYMappings(self):
        if self.trainY.dtype.kind in {'U', 'S'}: # a string
            i = 0
            for value in np.unique(self.trainY):
                if value != None and value.strip() != "":
                    self.trainYMappingsStrToNum[value] = i
                    self.trainYMappingsNumToStr[i] = value
                    i = i + 1

    def _convertTrainYToNumeric(self):
        if self.trainY.dtype.kind in {'U', 'S'}: # a string
            self.trainY = np.vectorize(self.trainYMappingsStrToNum.get)(self.trainY)

    def _dropNATrainYRows(self):
        rowsToDrop = []
        for i in range(0,len(self.trainY)):
            rowsToDrop.append(i) if np.isnan(self.trainY[i]) else None
        self._dropRowsFromDFAndTrainY(rowsToDrop)

    def _dropRowsFromDFAndTrainY(self, rowsToDrop):
        self.df = self.df.drop(self.df.index[rowsToDrop])
        mask = np.ones(len(self.trainY), np.bool)
        mask[rowsToDrop] = 0
        self.trainY = self.trainY[mask]

    ########## Column Metadata ##########

    def _cleanColumnNamesDF(self):
        self.df.columns = self.df.columns.str.strip().str.lower().str.replace(' ', '_')

    def _setColumnDataTypes(self):
        columns = self.df.columns.values.tolist()
        for column in columns:
            if column in self.indexColumns or column in self.skipColumns:
                continue
            elif self.df[column].dtype == 'int64' or self.df[column].dtype == 'float64':
                self.numberColumns.append(column)
            elif self.df[column].dtype == 'object':
                self.categoryColumns.append(column)
            else:
                self.datetimeColumns.append(column)
                self.numberColumns.append(column)

    ########## Index ##########

    def _dropDuplicatesAndMissingRowsIfIndexIsSpecified(self):
        rowsToDrop = []
        if self.indexColumns != []:
            self.df['__trainY__'] = self.trainY
            self.df = self.df.drop_duplicates(subset=self.indexColumns)
            self.trainY = self.df['__trainY__'].values
            self.df = self.df.drop(['__trainY__'], 1)

            for i, row in self.df.iterrows():
                for column in self.indexColumns:
                    if ((self.df[column].dtype == 'int64' or self.df[column].dtype == 'float64') and (np.isnan(row[column]) or np.isinf(row[column]))) or row[column] == None:
                        rowsToDrop.append(i)
        self._dropRowsFromDFAndTrainY(rowsToDrop)

    ########## Datetimes ##########

    def _convertDatetimeToNumber(self):
        for column in self.datetimeColumns:
            values = []
            for i, row in self.df.iterrows():
                values.append((pd.datetime.now() - row[column]).days)
            self.df[column] = values

    ########## Numbers ##########

    def _saveMediansAndBounds(self):
        firstQuantiles = self.df.quantile(.25)
        thirdQuantiles = self.df.quantile(.75)

        self.medians = self.df.quantile(.50)
        self.lowerBounds = {}
        self.upperBounds = {}
        for column in self.numberColumns:
            self.lowerBounds[column] = self.medians[column] - 2*(self.medians[column] - firstQuantiles[column])
            self.upperBounds[column] = self.medians[column] + 2*(thirdQuantiles[column] - self.medians[column])

    def _fixMissingNumValuesAndInfinity(self):
        self.df = self.df.fillna(self.medians) # optionally: replace self.medians with 0
        self.df.replace([np.inf, -np.inf], np.nan)
        self.df = self.df.fillna(self.upperBounds)

    def _fixHighLeveragePoints(self):
        for i, row in self.df.iterrows():
            for column in self.numberColumns:
                if row[column] > self.upperBounds[column]:
                    self.df.at[i, column] = self.upperBounds[column]
                if row[column] < self.lowerBounds[column]:
                    self.df.at[i, column] = self.lowerBounds[column]

    ########## Categories ##########

    def _saveUniqueCategoryValues(self):
        for column in self.categoryColumns:
            self.uniqueCategoryValues[column] = []
            for value in self.df[column].unique():
                if value == None:
                    continue
                self.uniqueCategoryValues[column].append(value)
            self.uniqueCategoryValues[column].append('_Other')

    def _saveCategoryFrequenciesAndValuesThatDontMapTo_Other(self):
        for column in self.categoryColumns:
            _otherFrequency = 0
            self.valuesThatDontMapTo_Other[column] = ['_Other']
            frequencyPercentage = pd.value_counts(self.df[column].values, sort=False, normalize=True)
            self.categoryFrequencies[column] = {}
            for value in self.uniqueCategoryValues[column]:
                if value == '_Other':
                    continue
                elif frequencyPercentage[value] < .05:
                    _otherFrequency = _otherFrequency + frequencyPercentage[value]
                else:
                    self.valuesThatDontMapTo_Other[column].append(value)
                    self.categoryFrequencies[column][value] = frequencyPercentage[value]
            self.categoryFrequencies[column]['_Other'] = _otherFrequency

    def _dropCategoryColumnsWithAllMissingValues(self):
        columnsToRemove = []
        for column in self.categoryColumns:
            if len(self.uniqueCategoryValues[column]) == 1 and self.uniqueCategoryValues[column][0] == '_Other':
                columnsToRemove.append(column)
                self.categoryColumns.remove(column)
                self.columnsDropped.append(column)
        self.df = self.df.drop(columnsToRemove, 1)


    def _fixMissingCategoryValuesAndMapValuesTo_Other(self):
        for i, row in self.df.iterrows():
            for column in self.categoryColumns:
                if row[column] == None:
                    self.df.at[i, column] = self._getRandomCategoryBasedOnFrequencies(column)
                elif row[column] not in self.valuesThatDontMapTo_Other[column]:
                    self.df.at[i, column] = '_Other'

    def _getRandomCategoryBasedOnFrequencies(self, column):
        chosenValue, prevValue, cumulativeProbability = None, None, 0
        randomNumber = np.random.uniform(0,1,1)[0]
        for value in self.uniqueCategoryValues[column]:
            if value in self.valuesThatDontMapTo_Other[column]:
                probabilityOfValue, prevValue = self.categoryFrequencies[column][value], value
                cumulativeProbability = cumulativeProbability + probabilityOfValue
                if cumulativeProbability > randomNumber:
                    chosenValue = value
                    break
        return prevValue if chosenValue == None else chosenValue

    def _applyOneHotEncoding(self): # don't drop_first => one hot encoding instead of dummy encoding
        for column in self.categoryColumns:
            self.df = pd.concat([self.df.drop(column, axis=1), pd.get_dummies(self.df[column], prefix=column+"_", drop_first=False)], axis=1)

    ########## Class Imbalance ##########

    def _saveTrainYFrequencies(self):
        self.trainYFrequencies = pd.value_counts(self.trainY, sort=True, normalize=False)

    def _saveTrainYUpsamplesNeeded(self):
        maxValue = None
        for value in self.trainYMappingsStrToNum.values():
            frequency = self.trainYFrequencies[value]
            if maxValue == None or frequency > maxValue:
                maxValue = frequency

        minValue = ceil(maxValue / 2)

        for value in self.trainYMappingsStrToNum.values():
            actualFrequency = self.trainYFrequencies[value]
            idealTrainYFrequency = minValue if actualFrequency < minValue else actualFrequency
            self.trainYUpsamplesNeeded[value] = idealTrainYFrequency - actualFrequency

    def _fixTrainYImbalance(self):
        self.df['__trainY__'] = self.trainY
        for value in self.trainYMappingsStrToNum.values():
            samplesToGet = self.trainYUpsamplesNeeded[value]
            if samplesToGet > 0:
                upsampleRows = resample(self.df[self.df['__trainY__']==value],
                                    replace=True,
                                    n_samples=samplesToGet,
                                    random_state=123)
                self.df = pd.concat([self.df, upsampleRows])
        self.trainY = self.df['__trainY__'].values
        self.df = self.df.drop(['__trainY__'], 1)

    ########## Index ##########

    def _addIndex(self):
        indexColumns = []
        self.df['_id'] = np.arange(1,len(self.df.index)+1)
        if self.indexColumns != []:
            indexColumns = list(self.indexColumns)
        indexColumns.append('_id')
        self.df = self.df.set_index(indexColumns)

    ########## Get Final Column Names ##########

    def _saveFinalColumnNames(self):
        self.finalColumnNames = list(self.df)

    ########## New Data ##########

    def _newDataDropDroppedColumns(self):
        self.df = self.df.drop(self.columnsDropped, axis=1)

    def _newDataAddMissingFinalColumnNames(self):
        # Assuming only category columns will be missing
        for column in self.finalColumnNames:
            if column not in list(self.df):
                self.df[column] = np.zeros((len(self.df.index),1))

    def _newDataDropExtraColumnNames(self): # This hopefully does nothing - using it anyway
        columnsToDrop = []
        for column in list(self.df):
            if column not in self.finalColumnNames:
                columnsToDrop.append(column)
        self.df = self.df.drop(columnsToDrop, axis=1)

### throwaway test:
# df = pd.DataFrame({'col2': [None,None,None,9,5,10,11,12,13,14,None,None,None,9,5,10,11,12,13,14,11,12,13,14,None,None,None,9,5,10,11,12,13,14,None,None,None,9,5,10,11,12,13,14,11,12,13,14]
#                   , 'col3': ['test1','test1','test1','test3',None,None,'test1','test1','test2','test2','test1','test1','test1','test1',None,None,'test1','test1','test2','test2', 'test1','test1','test2','test2','test1','test1','test1','test1',None,None,'test1','test1','test2','test2','test1','test1','test1','test1',None,None,'test1','test1','test2','test2', 'test1','test1','test2','test2']
#                   , 'col4': [None, 5, 3 ,6 ,8, 9, 14, 87, 999 ,9999,None, 5, 3 ,6 ,8, 9, 14, 87, 999 ,9999, 14, 87, 999 ,9999,None, 5, 3 ,6 ,8, 9, 14, 87, 999 ,9999,None, 5, 3 ,6 ,8, 9, 14, 87, 999 ,9999, 14, 87, 999 ,9999]
#                   , 'col5': ['a','a',None,None,'adsf','bas',None,None,None,None,None,None,None,None,'adsf','bas',None,None,None,None,None,None,None,None,'a','a',None,None,'adsf','bas',None,None,None,None,None,None,None,None,'adsf','bas',None,None,None,None,None,None,None,None]})
#
# targetY = ['a','b','c','a','a','g','b','a','i','t','a','b','c','a','a','g','b','a','i','t','b','a','i','t','a','b','c','a','a','g','b','a','i','t','a','b','c','a','a','g','b','a','i','t','b','a','i','t']
# indexColumns = ['col4']
#
# neat = Neat(df, targetY, indexColumns)
#
# print(neat.df)
# #df = neat.df
#
# neat.cleanNewData(df)
#
# neat.df
