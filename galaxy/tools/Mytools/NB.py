import sys
import header
'''

Loading the Dataset HERE ---

'''
inputFile = sys.argv[1]
train_data, train_target,test_data,test_target,k = header.getTrainAndTestData(inputFile)

header.NaiveBayes(train_data, train_target,test_data,test_target,k)