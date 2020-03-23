import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(1, '../fashion_mnist/utils')
import os
import time
import pickle as pkl

import mnist_reader
from init import initializeMNIST as mnist

import keras
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout, Flatten,Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
#from keras.layers.normalization import BatchNormalization

class CNN():
	def __init__(self,mnist):
		self.mnist=mnist
		self.batch_size=512
		self.epochs=20
		self.mnist.normalize_data()
		self.mnist.reshape_data()
		self.model=self.create_nn()
		self.loss=[]
		self.accuracy=[]
		self.hist=[]
		self.compile_model()
		self.load_model()
		
		
	def load_model(self):
		if os.path.isfile("../Datadumps/model.h5") and os.path.isfile("../Datadumps/history.pkl"):
			print("Loading existing model...")
			self.model = load_model('../Datadumps/model.h5')
			self.history = pkl.load(open('../Datadumps/history.pkl','rb'))
			print(self.history.history)
		else:	
			print("Creating new model...")
			self.history=self.train_model()
			#print(self.history.history)
			self.save('../Datadumps/model.h5')
			pkl.dump(self.history,open('../Datadumps/history.pkl','wb'))


	def create_nn(self):
		model = Sequential()
		model.add(Conv2D(filters=32,kernel_size=3,input_shape = self.mnist.x_train[0].shape))
		model.add(Activation('relu'))
		model.add(MaxPooling2D(pool_size=2))
		model.add(Dropout(0.2))
		model.add(Flatten())
		model.add(Dense(32))
		model.add(Activation('relu'))
		model.add(Dense(10))
		model.add(Activation('softmax'))
		return model
	
	def compile_model(self):
		self.model.compile(loss ='sparse_categorical_crossentropy', optimizer=Adam(lr=0.001),metrics =['accuracy'])

	def model_summary(self):
		print(self.model.summary())

	def train_model(self):
		history = self.model.fit(self.mnist.x_train, self.mnist.y_train,
			batch_size=self.batch_size,
			epochs=self.epochs,
			verbose=1)
		self.hist.append(history)
		return history
		

	def evaluate_model(self):
		score = self.model.evaluate(self.mnist.x_test,self.mnist.y_test,verbose=0)
		print(f'Test Loss : {score[0]:.4f}')
		print(f'Test Accuracy : {score[1]:.4f}')

	def save(self, name):
		self.model.save(name)
	
	def plot_model(self):
		accuracy = self.history.history['acc']

		loss = self.history.history['loss']
		
		epochs = range(len(accuracy))
		plt.plot(epochs, accuracy, 'bo', label='Training Accuracy')
		plt.title('Training accuracy')
		plt.figure()
		plt.plot(epochs, loss, 'bo', label='Training Loss')
		plt.title('Training loss')
		plt.show()


from keras import backend as K
import tensorflow as tf
if __name__=='__main__':
	model=CNN(mnist())
	model.model_summary()
	model.evaluate_model()
	model.plot_model()