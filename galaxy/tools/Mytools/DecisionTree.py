import sys
from header import getTrainAndTestData, DecisionTree
'''

Loading the Dataset HERE ---

'''

inputFile = sys.argv[1]
train_data, train_target,test_data,test_target,k = getTrainAndTestData(inputFile)

DecisionTree(train_data, train_target,test_data,test_target,k)




