import numpy as np
import csv
import sys
import os
from numpy import linalg
from collections import Counter
import time
import matplotlib.pyplot as plt
import seaborn as sn

configFile = sys.argv[1]
fileNameTrain = sys.argv[2]
fileNameTest = sys.argv[3]
try:
	partNum = sys.argv[4]
except: partNum = "b"


def accuracy(actual, predicted):
	if(len(actual) != len(predicted)):
		return 0
	count =0
	for a,p in zip(actual, predicted) :
		if(a==p):
			count += 1
	return 100.0 * float(count)/float(len(predicted))



# reading parameters from configFile
with open(configFile,'r') as f:
	inputLayerSize = int(f.readline())
	outputLayerSize = int(f.readline())
	batchSize = int(f.readline())
	hiddenLayerSize = int(f.readline()) + 1
	neuronsArray = np.array([inputLayerSize])
	neuronsArray = np.append(neuronsArray, f.readline().split(" ")).astype(np.int)
	neuronsArray = np.append(neuronsArray,outputLayerSize)
	nonlinearity = f.readline().rstrip('\n')
	learningRate = f.readline().rstrip('\n')
	weights = []
	biases = []
	for j, k in zip(neuronsArray[:-1], neuronsArray[1:]):
		warr = np.random.uniform(-0.5, 0.5, (k,j))
		barr = np.random.uniform(-0.5, 0.5,(k,1))
		weights.append(warr)
		biases.append(barr)


	

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
	return trainData, testData


trainData, testData = readData(fileNameTrain, fileNameTest)
trainDataX = trainData[:,0:85]
testDataX = testData[:,0:85]
trainDataY = trainData[:,85:]
testDataY = testData[:,85:]
'''weights =[]
biases = []
weights.append(np.array([[0.15,0.2],[0.25,0.3]]))
weights.append(np.array([[0.4,0.45],[0.5,0.55]]))
biases.append(np.array([[0.35],[0.35]]))
biases.append(np.array([[0.3],[0.3]]))       
neuronsArray = [2,2,2]

trainDataX = np.array([[0.05 ,0.1]])     
trainDataY = np.array([[0.01, 0.99]])'''



def dsigmoid(x):
	dsig = x * (1.0 -x)
	return dsig

def relu(x):
	y = (x>0)
	return x * y.astype(np.int)

def drelu(x):
	y = (x>0)
	return y.astype(np.int)



def quadcost(a, y):
	return 0.5 * (linalg.norm(a - y) ** 2)


def sig(x):
	return 1.0 / (1.0 + np.exp(-x)) if x>0 else 1 - 1.0 / (1.0 + np.exp(x))

def sigmoid(x):
	vec_sig = np.vectorize(sig)
	return vec_sig(x)

def total_error(x,y):
	feedForward = forwardProp(x)
	out = 0.5 * np.linalg.norm(feedForward[-1] - y.T) ** 2
	
	return out/len(x)


def forwardProp(a):
	a = a.T
	outputs = []
	outputs.append(a)
	batch_size = a.shape[1]
	# Feed the data forward
	for j in range(hiddenLayerSize):

		w = weights[j]
		b = np.repeat(biases[j],batch_size, axis = 1)
		net_j = np.dot(w , a) + b 
		# ReLU is only used in hidden layers
		if nonlinearity == "relu" and j < hiddenLayerSize - 1:
			a = relu(net_j)
		else:
			a = sigmoid(net_j)

		outputs.append(a)

	return outputs


def back_propagation(outputs, target,eta):

	batch_size = target.shape[0]

	# Computation is done moving backwards
	for j in range(1, hiddenLayerSize +1):
		# ReLU is only used in hidden layers
		if nonlinearity == "relu" and -j != -1:
			del_out = drelu(outputs[-j])
		else:
			# del_Oj / del_Netj = Oj (1 - Oj)
			del_out = dsigmoid(outputs[-j])

		# At the last layer
		if j == 1:
			delta = del_out * (outputs[-1] - target.T)
		else:
			delta = del_out * np.dot(weights[-j + 1].T , delta)

		# Gradients at this layer
		weights[-j] -= eta * np.dot(delta, outputs[-j - 1].T) / batch_size
		b = np.sum(delta, axis=1) / batch_size
		b = np.reshape(b, (b.shape[0], 1))
		biases[-j] -= eta * b
		

	#return dw, db


def predict(X, y, returnpred = False):
	"""Predict classes for data."""

	# Otherwise use the index of the neuron with maximum output
	predicted = forwardProp(X)
	predicted = np.argmax(predicted[-1], axis = 0)
	
	y = np.argmax(y, axis =1)
	if returnpred:
		return accuracy(y, predicted), predicted
	else:
		return accuracy(y, predicted)

def trainNN(eta = 0.1, epochs = 100, tol = 10**-10):
	idx = np.arange(len(trainDataX))
	epoch = 0
	error = np.inf
	valid_error = np.inf
	layersize = hiddenLayerSize + 1
	
	lrFlag = False
	
	while True:
		if eta <= 0:
			print("Zero Learning rate : " ,eta)
			break
		epoch += 1
		#np.random.shuffle(idx)
		for i in range(0, len(trainDataX), batchSize):
			batch = idx[i:i + batchSize]
			Xb, yb = trainDataX[batch], trainDataY[batch]
			
			layer_outputs = forwardProp(Xb)
			back_propagation(layer_outputs, yb, eta)
		error_old = error
		
	
		error = total_error(trainDataX, trainDataY)
		print("Error epoch" , epoch , error)
		if(epoch % 20 == 0):
			print("acc" ,predict(trainDataX, trainDataY), error)
		
		if learningRate == "variable":
			if not lrFlag and (error - error_old) < 10** -10 :
				lrFlag = True
			elif lrFlag and (error - error_old) < 10** -10 :
				eta = float(eta) / 5.0
				lrFlag = False
				print("Eta value updated : " , eta)
			else:
				lrFlag = False
		
		
		# Early stopping
		if epoch == epochs:
			print("\nMaximum epochs reached")
			break
		elif learningRate == "fixed" and abs(error_old - error) <= tol:
			print("Error threshold reached")
			break


def plot(x, accuracies , plotName, xlabel = 'Number of nodes'):
	line1, = plt.plot(x, accuracies["train"],  "b", label ="Training accuracy ")
	line2, = plt.plot(x, accuracies["test"], "r", label ="Testing accuracy ")

	plt.legend(handles = [line1, line2 ])
	plt.ylabel('Accuracy')
	plt.xlabel(xlabel)
	OUT = "output/"
	plt.title(plotName)
	# creating output dorectory if not exist

	if not os.path.exists(OUT):
		os.makedirs(OUT)
	figure = os.path.join(OUT, plotName + ".png")

	plt.savefig(figure)
	#plt.close()
	plt.show()
	plt.close()
	
	return


#Plot Confusion Matrix for testing and Training Accuracies
def plotConfusionMatrix(actuallabels, predictedlabels, neuron, fname):
	ratings = list(sorted(set(actuallabels)))
	confusionMatrix = {}
	for cls in ratings:
		confusionMatrix[cls] = {cc: 0 for cc in ratings}
	for (actual ,predicted) in zip(actuallabels , predictedlabels) :
		confusionMatrix[actual][predicted] += 1
	arrayMatrix = []
	for cls in sorted(confusionMatrix.keys()):
		tempArr = []
		for c in sorted(confusionMatrix[cls].keys()):
			tempArr.append(confusionMatrix[cls][c])
		arrayMatrix.append(tempArr)
	
	plt.figure(figsize=(10, 7))

	ax = sn.heatmap(arrayMatrix, fmt="d", annot=True, cbar=False,
                    cmap=sn.cubehelix_palette(15),
                    xticklabels=ratings, yticklabels=ratings)
	# Move X-Axis to top
	ax.xaxis.tick_top()
	ax.xaxis.set_label_position('top')
	
	ax.set(xlabel="Predicted", ylabel="Actual")
	
	OUT = "output/"
	if not os.path.exists(OUT):
		os.makedirs(OUT)
	figure = os.path.join(OUT, fname + "neuron " + str(neuron) + ".jpg")

	plt.title(fname + " with neuron = " + str(neuron) , y = 1.08 , loc = "center")
	plt.savefig(figure)
	plt.show()
	plt.close()
	


if partNum == "b" :
	trainNN(epochs = 1500, tol = 10** -16, eta = 0.1)
	print("Accuracy On training :" , predict(trainDataX, trainDataY))
	print("Accuracy On testing :" , predict(testDataX, testDataY))

elif partNum == "c" or partNum == "d" :
	accuracies = {"train" : [], "test" : []}
	
	neuron = [5, 10, 15, 20, 25]
	test_y = np.argmax(testDataY, axis =1)
	train_y = np.argmax(trainDataY, axis =1)
	for j in neuron:
		neuronsArray = np.array([]).astype(np.int)
		if partNum == "c":
			neuronsArray = np.append(neuronsArray, [int(inputLayerSize), int(j), int(outputLayerSize)])
		else:
			neuronsArray = np.append(neuronsArray, [int(inputLayerSize),int(j), int(j), int(outputLayerSize)])
		weights = []
		biases = []
		for j, k in zip(neuronsArray[:-1], neuronsArray[1:]):
			warr = np.random.uniform(-0.5, 0.5,(k,j))
			barr = np.random.uniform(-0.5,0.5,(k,1))
			weights.append(warr)
			biases.append(barr)
		#weights = np.array(weights)
		#biases = np.array(biases)
		print(neuronsArray)
		start_time = time.time()
	
		trainNN(epochs = 2500, tol = 10** -20, eta = 0.1)
		print("---Time for training with neurons" + str(j) + " %s seconds ---" % (time.time() - start_time))
		#print(Counter(trainDataY))
		acc = predict(trainDataX, trainDataY)
		accuracies["train"].append(acc)
		print("Accuracy On training :" , acc)
		acc, predictedlabels = predict(testDataX, testDataY, returnpred = True)
		accuracies["test"].append(acc)
		print("Accuracy On testing :" , acc)
		plotConfusionMatrix(test_y, predictedlabels, j, "confusionMatrix_" + partNum)
	plot(neuron, accuracies , "neurons_vs_accuracies_hidden_" + partNum, xlabel = 'Number of neurons')
	print(neuron, accuracies)

elif partNum == "e" or partNum =="f" :
	#accuracies = {"train" : [], "test" : []}
	
	neuron = [5, 10, 15, 20, 25]
	test_y = np.argmax(testDataY, axis =1)
	train_y = np.argmax(trainDataY, axis =1)
	for i in range(2):
		hiddenLayerSize = i+2
		accuracies = {"train" : [], "test" : []}
		for j in neuron:
			neuronsArray = np.array([]).astype(np.int)
			if i == 0:
				neuronsArray = np.append(neuronsArray, [int(inputLayerSize), int(j), int(outputLayerSize)])
			else:
				neuronsArray = np.append(neuronsArray, [int(inputLayerSize),int(j), int(j), int(outputLayerSize)])
			
			weights = []
			biases = []
			for j, k in zip(neuronsArray[:-1], neuronsArray[1:]):
				warr = np.random.uniform(-0.5, 0.5,(k,j))
				barr = np.random.uniform(-0.5,0.5,(k,1))
				weights.append(warr)
				biases.append(barr)
			#weights = np.array(weights)
			#biases = np.array(biases)
			print(neuronsArray)
			start_time = time.time()
		
			trainNN(epochs = 2500, tol = 10** -25, eta = 0.1)
			print("---Time for training with neurons" + str(j) + " %s seconds ---" % (time.time() - start_time))
			#print(Counter(trainDataY))
			acc = predict(trainDataX, trainDataY)
			accuracies["train"].append(acc)
			print("Accuracy On training :" , acc)
			acc, predictedlabels = predict(testDataX, testDataY, returnpred = True)
			accuracies["test"].append(acc)
			print("Accuracy On testing :" , acc)
			plotConfusionMatrix(test_y, predictedlabels, j, "confusionMatrix_" + partNum + "_i" + str(i))
		plot(neuron, accuracies , "neurons_vs_accuracies_hidden_" + partNum + "_i" + str(i), xlabel = 'Number of neurons')
		print(neuron, accuracies)

	
		
		
	
	





#trainNN(eta = 0.1, epochs=1000 , tol = 10**-10)
#print("acc" ,predict(trainDataX, trainDataY))



	

