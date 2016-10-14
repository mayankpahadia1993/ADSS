from sklearn.datasets import load_iris
from sklearn import tree
from numpy import genfromtxt
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm 
from sklearn.cluster import KMeans 
from numpy import isnan
from collections import Counter
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import random
import sys

def printCM(cm,k):
    for i in range(k):
        print i,'-',
        for j in range(k):
            print cm[i,j],
        print ''

def printClassBasedAccuracy(cm,k):
    s=0
    
    for i in range(k):
        s=0
        for j in range(k):
            s+=cm[j,i]
        if s!=0:
            print i,cm[i,i]*100.0/s
        else:
            print i,0

def calculateCM(a,test_target,k):
    count=0
    for i in range(len(a)):
        if test_target[i][0]== a[i]:
            count+=1
    cm = {}
    for i in range(k):
        for j in range(k):
            cm[i,j]=0
    for i in range(len(a)):
        cm[a[i],test_target[i][0]]+=1
    print "Number of correctly labeled points out of a total %d points : %d" % (len(test_target),count)
    accuracy = float(count) / len(test_target)
    accuracy *= 100
    print accuracy 
    return cm


def Kmeans(train_data, train_target,test_data,test_target,k):
    klf = KMeans(n_clusters=2,init='random')
    klf = klf.fit(train_data)
    kpred = klf.predict(test_data)
    kcm = calculateCM(kpred,test_target,k)
    print "kmeans"
    printCM(kcm,k)
    printClassBasedAccuracy(kcm,k)




'''

Loading the Dataset HERE ---

'''

inputFile = sys.argv[1]
print 'works till here'

x= genfromtxt(inputFile, delimiter=',')
# randomly take 40% as test data from training data
random.shuffle(x)

s = len(x)
s = s * 60
s = s/ 100
train_data= x[:s,:]
test_data = x[s:,:]
#train_data = genfromtxt('BreakTraining.csv', delimiter=',')

#print my_data
#print train_data[:10]
whereNan = isnan(train_data)
train_data[whereNan] = 0

train_target = train_data[:,-1:]#last column is class
#print my_target
train_data = train_data[:,:-1]
#print my_data




whereNan = isnan(test_data)
test_data[whereNan] = 0
#print test_data[:10]

#print my_data
test_target = test_data[:,-1:]
#print my_target
test_data = test_data[:,:-1]
#print my_data
a = set(train_target.flatten())
k=len(a)#number of classes

Kmeans(train_data, train_target,test_data,test_target,k)

Kmeans(train_data, train_target,train_data, train_target,k)




