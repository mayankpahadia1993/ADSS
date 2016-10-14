import sys
from header import getTrainAndTestData, Kmeans
'''

Loading the Dataset HERE ---

'''

inputFile = sys.argv[1]
train_data, train_target,test_data,test_target,k = getTrainAndTestData(inputFile)

Kmeans(train_data, train_target,test_data,test_target,k)

Kmeans(train_data, train_target,train_data, train_target,k)




