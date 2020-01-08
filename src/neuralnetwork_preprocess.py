import numpy as np
import csv
import sys

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

categoricalAttribute = [0,1,2,3,4,5,6,7,8,9,10]

fileNameTrain = sys.argv[1]
fileNameTest = sys.argv[2]

fileNameTrainWrite = sys.argv[3]
fileNameTestWrite = sys.argv[4]

def readData(filenametrain, filenametest, delimeter = ","):
	with open(filenametrain ,'r') as f:
		reader = csv.reader(f, delimiter=delimeter)
		# get all the rows as a list
		data = list(reader)
		# transform data into numpy array
		trainData = np.array(data).astype(np.int)
	print(trainData.shape)
	with open(filenametest ,'r') as f:
		reader = csv.reader(f, delimiter=delimeter)
		# get all the rows as a list
		data = list(reader)
		# transform data into numpy array
		testData = np.array(data).astype(np.int)
	print(testData.shape)
	
	m_train , n = trainData.shape
	#trainY = trainData[:,n-1].reshape(m_train,1)
	m_test , _ = testData.shape
	#testY = testData[:,n-1].reshape(m_test,1)
	
	#trainX = trainData[:,0:n-1]
	#testX = testData[:,0:n-1]
	concateData = np.concatenate((trainData, testData))
	for j in categoricalAttribute:
		data = concateData[:,j]
		label_encoder = LabelEncoder()
		integer_encoded = label_encoder.fit_transform(data)
		onehot_encoder = OneHotEncoder(sparse=False)
		integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
		onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
		concateData = np.concatenate((concateData, onehot_encoded), axis = 1)
	concateData = np.delete(concateData,categoricalAttribute, 1)
	concateData = np.array(concateData).astype(np.int)
	#print(concateData[0])
	#trainData = np.concatenate((concateData[0:m_train], trainY), axis = 1)
	#testData = np.concatenate((concateData[m_train:m_train+m_test], testY), axis = 1)
	trainData = concateData[0:m_train]
	testData = concateData[m_train : m_train+m_test]
	return trainData, testData


trainData, testData = readData(fileNameTrain, fileNameTest)
print(trainData.shape, testData.shape)
np.savetxt(fileNameTrainWrite, trainData, fmt='%i',delimiter=",")
np.savetxt(fileNameTestWrite, testData, fmt='%i',delimiter=",")





