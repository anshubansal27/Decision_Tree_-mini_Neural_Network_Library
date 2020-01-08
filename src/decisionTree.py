import numpy as np
import csv
import sys
from anytree import Node, RenderTree, PreOrderIter
from collections import Counter 
from collections import deque
from tqdm import tqdm
import os

from sklearn.ensemble import RandomForestClassifier

from functools import partial

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import ParameterGrid

from matplotlib import pyplot as plt

import warnings
warnings.filterwarnings('ignore')

fileNameTrain = sys.argv[1]
fileNamevalidation = sys.argv[2]
fileNameTest = sys.argv[3]
binarize = 0
partNum = sys.argv[4]
if(partNum == "a" or partNum == "b"):
	binarize = 1

continuousAttribute = [1,5,12,13,14,15,16,17,18,19,20,21,22,23]
categoricalAttribute = [3,4,6,7,8,9,10,11]
attributes = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]

# reading data from file for label 2 and 3 and preprocessing of data
def readData(filename, delimeter = ","):
	with open(filename, 'r') as f:
		reader = csv.reader(f, delimiter=delimeter)
		# get header from first row
		headers = next(reader)
		headers = next(reader)
		# get all the rows as a list
		data = list(reader)
		# transform data into numpy array
		data = np.array(data).astype(np.int)
	m,n = data.shape
	print(data.shape)
	
	if(binarize):
		# preprocessing of continuous data 
		median = np.median(data, axis=0)
		#print(median)
		for i in range(m):
			for j in continuousAttribute:
				if data[i][j] >= median[j]:
					data[i][j] = 1
				else:
					data[i][j] = 0

	return data

trainData = readData(fileNameTrain)
validationData = readData(fileNamevalidation)
testData = readData(fileNameTest)

# calculating entropy according to the output
def entropy(Y):
	count = Counter(Y)
	probabilities = [float(count[c]) / float(len(Y)) for c in count]
	ent = sum(p * np.log2(p) for p in probabilities if p)		
	return -1 * ent

def partition(data):
	#datanode = np.array([trainData[i] for i in dataindex ])
	#d = datanode.T
	#x = d[attribute]
	#outArr ={v: np.array([dataindex[i] for i in np.where(data == v)[0]]) for v in np.unique(x)}
	outArr ={v: np.where(data == v)[0] for v in np.unique(data)}
	#print(outArr)
	return outArr

# use information gain to find best attribute to split on
def findBestAttribute(data):
	medianFinal = None
	median = None
	bestgain = -1
	bestattribute = -1
	partitionFinal = {}
	Y = data[24] # last column is labels
	for j in attributes:
		Xa = data[j]
		median = None
		if j in continuousAttribute:
			if not binarize:
				median = np.median(Xa)
				Xa = (Xa >= median).astype(int)
		# Create partitions over this attribute
		#partitionData = partition(X)
		entropy_Y_X = 0
		#for p in partitionData.values():
		#	entropy_Y_X += (len(p) / len(X)) * entropy(y[p])
		#gain = entropy(y) - entropy_Y_X
		partition_j = partition(Xa)
		entropy_Y_Xa = sum((float(len(p)) / float(len(Xa))) * entropy(Y[p]) for p in partition_j.values())
		
		gain = entropy(Y) - entropy_Y_Xa
		#print(j, gain)

		if (gain > bestgain):
			bestgain = gain
			bestattribute = j
			partitionFinal = partition_j
			medianFinal = median
	return gain, bestattribute , partitionFinal, medianFinal

#d = trainData.T
#gain , bestAttribute = findBestAttribute(d)
#print(gain ,  bestAttribute)


def decisionTree(rootNode, data):
	d = data.T
	Y = d[24]
	nsamples = Counter(Y)
	for letter, count in nsamples.most_common(1):
			rootNode.name["class"] = letter
	if len(nsamples) <= 1:
		rootNode.name["splitatt"] = None
		rootNode.name["nsamples"] = nsamples
	else:
		# Find the attribute that maximizes the gain
		gain, split_attr, partition_attr, median = findBestAttribute(d)
		#print("test", split_attr)
		if len(partition_attr) > 1:
			rootNode.name["splitatt"] = split_attr
			rootNode.name["median"] = median
			for val, part in partition_attr.items():
				#print(val , part)
				node = Node({"splitval" : val, "median" : None}, parent = rootNode)
				decisionTree(node, data[part])
				
		else:
			
			rootNode.name["splitatt"] = None
			rootNode.name["nsamples"] = nsamples
def accuracy(actual, predicted):
	if(len(actual) != len(predicted)):
		return 0
	count =0
	for a,p in zip(actual, predicted) :
		if(a==p):
			count += 1
	return 100.0 * float(count)/float(len(predicted))

def predict(exp , rootNode):
	if rootNode.name["splitatt"] is None:
		return rootNode.name["class"]
	else:
		# get value of the test for split attribute
		val = exp[rootNode.name["splitatt"]]
		if not binarize and rootNode.name["median"] is not None:
			val = int(val >= rootNode.name["median"])
		
		# Decide which child to go next to based on split values
		children = {c.name["splitval"] : c for c in rootNode.children}
		child = children.get(val)
		# If there isn't a correct outgoing edge
		# then just return the majority class
		if not child:
			#print("in")
			return rootNode.name["class"]
		else:
			return predict(exp, child)

def prediction(testData, rootNode):
	test = testData.T
	actual = test[24]
	predicted = [predict(x, rootNode) for x in testData]
	acc =  accuracy(actual, predicted)
	return acc


def getnodes(rootNode):
	"""Iterate over all nodes in the tree in BFS order."""

	q = deque([rootNode])

	while q:
		node = q.popleft()
		q.extend(node.children)

		yield node
def node_count(rootNode):
	if not rootNode.children:
		return 1
	else:
		sumnode = 0
		for j in rootNode.children:
			sumnode = sumnode + node_count(j)  
		return 1 + sumnode


def post_prune(rootNode, train_data, test_data, valid_data):
	"""Prune by Brute-force - calculating accuracy before and after removing a node."""
	nodes = list(getnodes(rootNode))
	nodes.reverse()
	
	nodecounts = []
	accuracies = {"train": [], "test": [], "valid": []}

	# Iteate over all nodes and decide whether to keep this or not.
	for node in tqdm(nodes, ncols=80, ascii=True):

		# No point in checking a leaf node
		if node.children:

			# Accuracy before removing the node
			val_acc_before = prediction(valid_data, rootNode)

			# Remove the node
			children_backup = node.children
			node.children = ()

			# Accuracy before removing the node
			val_acc_after = prediction(valid_data, rootNode)

			# The tree remains pruned if accuracy increased afterwards
			if val_acc_after > val_acc_before:
				nodecounts.append(node_count(rootNode))

				accuracies["train"].append(prediction(train_data, rootNode))
				accuracies["test"].append(prediction(test_data, rootNode))
				accuracies["valid"].append(val_acc_after)

			elif val_acc_after < val_acc_before:
				# Add the node back if it accuracy wasn't improved
				node.children = children_backup
	
	nodecounts.reverse()
	for j, val in accuracies.items():
		val.reverse()

	return nodecounts, accuracies, rootNode


def removeNode(node):

        # Root can not be removed
        if node.parent is None:
            return

        node.parent = None

        #return node.parent


def computeAccuracies(rootNode, trainData, testData, validData, step = 100):
	nodes = list(getnodes(rootNode))
	nodes.reverse()
	
	totalnodes = len(nodes)
	
	nodecounts = []
	accuracies = {"train": [], "test": [], "valid": []}
	
	for i in tqdm(range(0, len(nodes), step), ncols=80, ascii=True):
		removenodes = nodes[i: i+step]
		for node in removenodes:
			removeNode(node)

		# Prevent total number nodes from going below 0
		# (this will happen at the last iteration)
		totalnodes = max(totalnodes - step, 0)
		nodecounts.append(totalnodes)

		accuracies["train"].append(prediction(trainData, rootNode))
		accuracies["test"].append(prediction(testData, rootNode))
		accuracies["valid"].append(prediction(validData, rootNode))
	return nodecounts , accuracies


def plot(nodeCount, accuracies , plotName, invert = False, xlabel = 'Number of nodes'):
	if xlabel =='Number of nodes':
		line1, = plt.plot(nodeCount, accuracies["train"],  "b", label ="Training accuracy :" + str(accuracies["train"][0]))
		line2, = plt.plot(nodeCount, accuracies["test"], "r", label ="Testing accuracy :" +str(accuracies["test"][0]))
		line3, = plt.plot(nodeCount, accuracies["valid"], "g", label ="Validation Accuracy :" +str( accuracies["valid"][0]))
	else:
		line1, = plt.plot(nodeCount, accuracies["train"],  "b", label ="Training accuracy ")
		line2, = plt.plot(nodeCount, accuracies["test"], "r", label ="Testing accuracy ")
		line3, = plt.plot(nodeCount, accuracies["valid"], "g", label ="Validation Accuracy ")


	if invert:
		plt.gca().invert_xaxis()
	plt.legend(handles = [line1, line2, line3])
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
	
	return


def findMultiPathAttributes(rootNode):
	def multipathattr(node,attr):
		if not node.children:
			return []
		else:
			mpa = partial(multipathattr, attr=attr)
			max_path_thresholds = max(map(mpa, node.children), key=len)

		if attr == node.name["splitatt"] and node.name["median"] is not None:
			return [node.name["median"]] + max_path_thresholds
		else:
			return list(max_path_thresholds)
	
	thresholdattr = {}
	for attr in continuousAttribute:
		thresholdattr[attr] = multipathattr(rootNode,attr)
	return thresholdattr

# default parameters already set here
def decisiontree_Skicit(criterion = "entropy", randomstate = 0 , max_depth = None, min_samples_split = 2, min_samples_leaf=1, validaccuracy = False):
	#print(criterion, randomstate, max_depth, min_samples_split, min_samples_leaf, validaccuracy)
	_, n = trainData.shape
	clf = DecisionTreeClassifier(criterion=criterion, random_state=randomstate, max_depth = max_depth,min_samples_split = min_samples_split,  min_samples_leaf = min_samples_leaf)
	clf.fit(trainData[:,1:n-1], trainData[:, n-1])
	
	
	testing_acc =None
	training_acc = None
	validation_acc = 100 * clf.score(validationData[:,1:n-1], validationData[:, n-1])
	if not validaccuracy:
		testing_acc = 100 * clf.score(testData[:,1:n-1], testData[:, n-1])
		training_acc = 100 * clf.score(trainData[:,1:n-1], trainData[:, n-1])
	
	
	return training_acc, validation_acc, testing_acc


# default parameters already set here
def randomForest_sklearn(criterion = "entropy", randomstate = 0 ,n_estimators = 10 ,max_features = "auto" ,max_depth = None, bootstrap = True):
	#print(criterion, randomstate, max_depth, min_samples_split, min_samples_leaf, validaccuracy)
	clf = RandomForestClassifier(criterion=criterion, random_state=randomstate, n_estimators= n_estimators ,max_features = max_features ,max_depth = max_depth, bootstrap = bootstrap)
	clf.fit(trainData[:,1:24], trainData[:, 24])
	
	validation_acc = 100 * clf.score(validationData[:,1:24], validationData[:, 24])
	testing_acc = 100 * clf.score(testData[:,1:24], testData[:, 24])
	training_acc = 100 * clf.score(trainData[:,1:24], trainData[:, 24])
	
	
	return training_acc, validation_acc, testing_acc



	

#plot([1,2,3,4,5], [33,33,4,4,6], "hi")
		

if partNum == "1" or partNum == "3":
	#plot([1,2,3,4], [22,33,44,55], "check")

	rootData = {"splitval" : None, "median" : None}
	rootNode = Node(rootData)
	decisionTree(rootNode , trainData)
	acc = prediction(trainData, rootNode)
	print("accuracy Training Data",acc)
	acc = prediction(testData, rootNode)
	print("accuracy Testing Data",acc)
	acc = prediction(validationData, rootNode)
	print("accuracy Validation Data",acc)
	if partNum == "c" :
		print("finding thresholds for multiple splits for continuos attributes")
		threshold = findMultiPathAttributes(rootNode)
		for att , value in threshold.items():
			if len(value) >1:
	
				print(att, value)
	nodecount, accuracies = computeAccuracies(rootNode, trainData, testData, validationData, step = 50)
	#print(nodecount, accuracies)
	plot(nodecount, accuracies , "part" + partNum)

elif partNum == "2":
	rootData = {"splitval" : None, "median" : None}
	rootNode = Node(rootData)
	decisionTree(rootNode , trainData)
	acc = prediction(trainData, rootNode)
	print("accuracy Training Data",acc)
	acc = prediction(testData, rootNode)
	print("accuracy Testing Data",acc)
	acc = prediction(validationData, rootNode)
	print("accuracy Validation Data",acc)
	nodeCounts, acc, rootNode = post_prune(rootNode, trainData, testData, validationData)
	#print(nodeCounts, acc)
	plot(nodeCounts, acc , "partB while pruning", invert = True)


elif partNum == "4":
	# with default parameters
	print("with default paramters:")
	train, valid, test = decisiontree_Skicit()
	print(" Accuracy on train : ", train)
	print(" Accuracy on valid : ", valid)
	print(" Accuracy on test : ", test)
	
	# with max_depth set
	print("Plotting the effect of max_depth vs accuracy over training, validation and testing data:")
	accuracies = {"train" : [], "valid" : [], "test" : []}
	depths = range(3,30)
	for d in depths:
		train, valid, test = decisiontree_Skicit(max_depth = d)
		accuracies["train"].append(train)
		accuracies["test"].append(test)
		accuracies["valid"].append(valid)
	plot(depths, accuracies , "max_depth_vs_accuracy_decisionTree",  xlabel = 'max_depths')

	# with min_samples_split
	print("Plotting the effect of min_samples_split vs accuracy over training, validation and testing data:")
	accuracies = {"train" : [], "valid" : [], "test" : []}
	splits = range(5, 200, 5)
	for d in splits:
		train, valid, test = decisiontree_Skicit(min_samples_split = d)
		accuracies["train"].append(train)
		accuracies["test"].append(test)
		accuracies["valid"].append(valid)
	plot(splits, accuracies , "min_samples_split_vs_accuracy_decisionTree",  xlabel = 'min_samples_split')
	
	# with min_samples_leaf
	print("Plotting the effect of min_samples_leaf vs accuracy over training, validation and testing data:")
	accuracies = {"train" : [], "valid" : [], "test" : []}
	splits = range(5, 200, 5)
	for d in splits:
		train, valid, test = decisiontree_Skicit(min_samples_leaf = d)
		accuracies["train"].append(train)
		accuracies["test"].append(test)
		accuracies["valid"].append(valid)
	plot(splits, accuracies , "min_samples_leaf_vs_accuracy_decisionTree",  xlabel = 'min_samples_split')

	print("\nNow running a grid search to find best parameters")

	
	results = {}
	# Range of parameters to find best accuracy over
	parameters = {'max_depth': range(5, 20),'min_samples_split': range(5, 100, 5),'min_samples_leaf': range(5, 100, 5),}

	# Run a custom search that scores on the validation set
	for param in ParameterGrid(parameters):
		train, valid, test = decisiontree_Skicit(**param)
		results[valid] = {'param' : param, "train" : train , "test" : test}
	print("Parameters resulting in max validation accuracy %r " % max(results), results[max(results)]['param'] )
	print( "Accuracy over training :", results[max(results)]['train'])
	print("Accuracy over testing :",  results[max(results)]['test'])

elif partNum == "5":
	#trainData = readData(fileNameTrain)
	#validationData = readData(fileNamevalidation)
	#testData = readData(fileNameTest)
	m_train , _ = trainData.shape
	trainY = trainData[:,24].reshape(m_train,1)
	m_test , _ = testData.shape
	testY = testData[:,24].reshape(m_test,1)
	m_valid , _ = validationData.shape
	validY = validationData[:,24].reshape(m_valid,1)
	trainX = trainData[:,0:24]
	testX = testData[:,0:24]
	validX = validationData[:,0:24]
	concateData = np.concatenate((trainX, testX, validX))
	for j in categoricalAttribute:
		data = concateData[:,j]
		label_encoder = LabelEncoder()
		integer_encoded = label_encoder.fit_transform(data)
		onehot_encoder = OneHotEncoder(sparse=False)
		integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
		onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
		concateData = np.concatenate((concateData, onehot_encoded), axis = 1)
	concateData = np.delete(concateData,categoricalAttribute, 1)
	trainData = np.concatenate((concateData[0:m_train], trainY), axis = 1)
	testData = np.concatenate((concateData[m_train:m_train+m_test], testY), axis = 1)
	validationData = np.concatenate((concateData[m_train+m_test:m_train+m_test+m_valid], validY), axis = 1)
	# parameters got from part d
	#{'min_samples_split': 95, 'max_depth': 19, 'min_samples_leaf': 85}
	train, valid, test = decisiontree_Skicit(min_samples_split = 95, max_depth = 19, min_samples_leaf = 85)
	print("with best parameters from part c")
	print( "Accuracy over training :", train)
	print( "Accuracy over Validation :", valid)
	print("Accuracy over testing :", test)


	# with default parameters
	print("with default paramters:")
	train, valid, test = decisiontree_Skicit()
	print(" Accuracy on train : ", train)
	print(" Accuracy on valid : ", valid)
	print(" Accuracy on test : ", test)
	
	# with max_depth set
	print("Plotting the effect of max_depth vs accuracy over training, validation and testing data:")
	accuracies = {"train" : [], "valid" : [], "test" : []}
	depths = range(3,30)
	for d in depths:
		train, valid, test = decisiontree_Skicit(max_depth = d)
		accuracies["train"].append(train)
		accuracies["test"].append(test)
		accuracies["valid"].append(valid)
	plot(depths, accuracies , "max_depth_vs_accuracy_decisionTree_e",  xlabel = 'max_depths')

	# with min_samples_split
	print("Plotting the effect of min_samples_split vs accuracy over training, validation and testing data:")
	accuracies = {"train" : [], "valid" : [], "test" : []}
	splits = range(5, 200, 5)
	for d in splits:
		train, valid, test = decisiontree_Skicit(min_samples_split = d)
		accuracies["train"].append(train)
		accuracies["test"].append(test)
		accuracies["valid"].append(valid)
	plot(splits, accuracies , "min_samples_split_vs_accuracy_decisionTree_e",  xlabel = 'min_samples_split')
	
	# with min_samples_leaf
	print("Plotting the effect of min_samples_leaf vs accuracy over training, validation and testing data:")
	accuracies = {"train" : [], "valid" : [], "test" : []}
	splits = range(5, 200, 5)
	for d in splits:
		train, valid, test = decisiontree_Skicit(min_samples_leaf = d)
		accuracies["train"].append(train)
		accuracies["test"].append(test)
		accuracies["valid"].append(valid)
	plot(splits, accuracies , "min_samples_leaf_vs_accuracy_decisionTree_e",  xlabel = 'min_samples_split')

	print("\nNow running a grid search to find best parameters")

	
	results = {}
	# Range of parameters to find best accuracy over
	parameters = {'max_depth': range(5, 20),'min_samples_split': range(5, 100, 5),'min_samples_leaf': range(5, 100, 5),}

	# Run a custom search that scores on the validation set
	for param in ParameterGrid(parameters):
		train, valid, test = decisiontree_Skicit(**param)
		results[valid] = {'param' : param, "train" : train , "test" : test}
	print("Parameters resulting in max validation accuracy %r " % max(results), results[max(results)]['param'] )
	print( "Accuracy over training :", results[max(results)]['train'])
	print("Accuracy over testing :",  results[max(results)]['test'])
	
		
		
	

elif partNum == "6":
	
	# with default parameters
	print("with default paramters: bootstrap = True")
	train, valid, test = randomForest_sklearn()
	print(" Accuracy on train : ", train)
	print(" Accuracy on valid : ", valid)
	print(" Accuracy on test : ", test)
	
	# with default parameters
	print("with default paramters: bootstrap = False")
	train, valid, test = randomForest_sklearn(bootstrap = False)
	print(" Accuracy on train : ", train)
	print(" Accuracy on valid : ", valid)
	print(" Accuracy on test : ", test)
	
	#n_estimators = 10 ,max_features = "auto" ,max_depth = None	
	
	# with n_estimators set
	print("Plotting the effect of n_estimators vs accuracy over training, validation and testing data:")
	accuracies = {"train" : [], "valid" : [], "test" : []}
	estimators = range(5,30)
	for d in estimators:
		train, valid, test = randomForest_sklearn(n_estimators = d)
		accuracies["train"].append(train)
		accuracies["test"].append(test)
		accuracies["valid"].append(valid)
	plot(estimators, accuracies , "n_estimators_vs_accuracy_randomForest",  xlabel = 'n_estimators')

	# with max_features
	print("Plotting the effect of max_features vs accuracy over training, validation and testing data:")
	accuracies = {"train" : [], "valid" : [], "test" : []}
	features = range(3, 20)
	for d in features:
		train, valid, test = randomForest_sklearn(max_features = d)
		accuracies["train"].append(train)
		accuracies["test"].append(test)
		accuracies["valid"].append(valid)
	plot(features, accuracies , "max_features_vs_accuracy_randomForest",  xlabel = 'max_features')
	
	# with max_depth
	print("Plotting the effect of max_depth vs accuracy over training, validation and testing data:")
	accuracies = {"train" : [], "valid" : [], "test" : []}
	depths = range(5,20)
	for d in depths:
		train, valid, test = randomForest_sklearn(max_depth = d)
		accuracies["train"].append(train)
		accuracies["test"].append(test)
		accuracies["valid"].append(valid)
	plot(depths, accuracies , "max_depth_vs_accuracy_randomForest",  xlabel = 'max_depth')

	print("\nNow running a grid search to find best parameters")

	
	results = {}
	# Range of parameters to find best accuracy over
	parameters = {'n_estimators': range(5, 20), 'max_features': range(3, 15), 'max_depth': range(8, 15),'bootstrap': [True, False],}

	# Run a custom search that scores on the validation set
	for param in ParameterGrid(parameters):
		train, valid, test = randomForest_sklearn(**param)
		results[valid] = {'param' : param, "train" : train , "test" : test}
	print("Parameters resulting in max validation accuracy %r " % max(results), results[max(results)]['param'] )
	print( "Accuracy over training :", results[max(results)]['train'])
	print("Accuracy over testing :",  results[max(results)]['test'])

else:

	print("Wrong Part Number")

	
	
	
	
		

