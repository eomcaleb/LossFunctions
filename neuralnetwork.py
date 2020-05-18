import numpy as np
import math

class NeuralNetwork(object):
	def __init__(self, ninput, epochs = 50, learning_rate=0.01, random_weights = True):
		self.epochs = epochs
		self.learning_rate = learning_rate
		self.ninput = ninput
		self.weights = np.random.rand(ninput + 1)

	def summationfunction(self, inputs):
		summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
		return summation	
		
	def predict(self, inputs):
		summation = np.dot(inputs, self.weights[1:]) + self.weights[0] > 0.0
		return summation	
		
	def bipolar_predict(self, inputs):
		summation = np.dot(inputs, self.weights[1:]) + self.weights[0] > 0.0
		summation = [x if (x == 1) else -1 for x in summation]
		return summation	

	def train(self, trainingdata, targets, verbose = False):
		trainingdata = np.array(trainingdata)
		# Regression Loss Functions - used to predict real values as output 
		mse_cost = []			# Mean Squared Error
		rmse_cost = []			# Root Mean Squared Error
		mae_cost = []			# Mean Absolute Error

		# Binary Classification Loss Function - used to predict binary values as output (0/-1 or 1)
		hingeloss_cost = []		# Hinge Loss Error
		logloss_cost = []		# Log Loss (a.k.a. Binary Cross Entropy) - Sigmoid (non-linear)
		eps = 1e-14
		
		for epoch in range(self.epochs): 
			summations = self.summationfunction(trainingdata)
			predicts = self.predict(trainingdata)
			bipolarsummations = self.bipolar_predict(trainingdata)	

			# We are using one optimization function here so it's not really using a specific loss function below
			errors = (targets - summations)
			self.weights[1:] += self.learning_rate * trainingdata.T.dot(errors)
			self.weights[0] += self.learning_rate * errors.sum()

			# Keep track of cost 
			mse_cost.append(np.sum((targets - summations) ** 2)/ self.ninput)
			rmse_cost.append(math.sqrt(np.sum((targets - summations) ** 2)/ self.ninput))
			mae_cost.append(np.sum(abs(targets - summations)) / self.ninput)

			hingeloss_cost.append(max(0 , (1 - np.sum(targets * bipolarsummations))) / self.ninput) 
			logloss_cost.append((-1 * np.sum(targets * np.log(np.clip(predicts,eps,1-eps)) + (1 - targets) * np.log(1 - np.clip(predicts,eps,1-eps))))/ self.ninput)

		# Cost Functions
		epoch = 1
		print ('Epoch\tMSE\tMAE\tRMSE\tHinge\tLog Loss')
		for mse, rmse, mae, hingeloss, logloss in zip(mse_cost, rmse_cost, mae_cost, hingeloss_cost, logloss_cost):
			print(epoch, '\t', '{0:.2f}'.format(mse), '\t', '{0:.2f}'.format(rmse), '\t', '{0:.2f}'.format(mae), '\t','{:.2f}'.format(hingeloss), '\t', '{0:.2f}'.format(logloss))
			epoch += 1
