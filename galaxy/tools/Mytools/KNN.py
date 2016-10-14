import sys
from header import getTrainAndTestData, knn
'''

Loading the Dataset HERE ---

'''

inputFile = sys.argv[1]
train_data, train_target,test_data,test_target,k = getTrainAndTestData(inputFile)

knn(train_data, train_target,test_data,test_target,k,3)




