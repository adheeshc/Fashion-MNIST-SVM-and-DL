import numpy as np
import matplotlib.pyplot as plt
import os
import time
import pickle as pkl

from sklearn import svm
from sklearn.metrics import accuracy_score,confusion_matrix

from init import initializeMNIST as mnist
from dim_red import myLDA,myPCA

class SVM():
	def __init__(self,tlink):
		self.mnist=mnist()
		self.x_train=self.mnist.x_train
		self.x_test=self.mnist.x_test
		self.y_train=self.mnist.y_train
		self.y_test=self.mnist.y_test

		self.tlink=tlink
		self.mlink="../Datadumps/"
		self.flink=self.mlink+self.tlink

		#myLDA(self.x_train,self.y_train,self.x_test,1)
		#myPCA(self.x_train,self.x_test,0.75)

		self.load_model()
		

	def load_model(self):

		if os.path.isfile(self.flink):
			print("Loading existing model...",end='')
			self.model=pkl.load(open(self.flink,"rb"))
			print('done\n')
		else:
			self.tic = time.time()
			print("Training new model...")
			self.model=self.train_model()
			self.train()
			print('DONE\n')

	def train_model(self):
		#LINEAR KERNEL SVM
		#model = svm.SVC(kernel='linear')
		
		#POLYNOMIAL KERNEL SVM
		#model = svm.SVC(kernel='poly', degree=2)
		#model = svm.SVC(kernel='poly', degree=3)
		model = svm.SVC(kernel='poly', degree=4)
		#model = svm.SVC(kernel='poly', degree=5)
		# model = svm.SVC(kernel='poly', degree=6)
		# model = svm.SVC(kernel='poly', degree=7)
		# model = svm.SVC(kernel='poly', degree=8)
		# model = svm.SVC(kernel='poly', degree=9)

		#GAUSSIAN KERNEL SVM
		# model = svm.SVC(kernel='rbf')

		#SIGMOID KERNEL SVM
		# model = svm.SVC(kernel='sigmoid')

		model.fit(self.x_train,self.y_train)
		return model

	def train(self):
		self.classifier_accuracy()
		self.predict()
		self.prediction_accuracy()
		self.confuse()
		self.toc = time.time()
		final=f"""classifier accuracy : {self.acc},
				predictions : {self.y_pred},
				prediction accuracy : {self.accuracy},
				confusion matrix : {self.cm},
				time taken : {self.toc-self.tic}"""
		pkl.dump(final,open(f'{self.tlink}','wb'))

	def classifier_accuracy(self):
		print('\nCalculating Accuracy of trained Classifier...',end='')
		self.acc = self.model.score(self.x_test,self.y_test)
		print('done')

	def predict(self):
		print('\nMaking Predictions on Test Data...',end='')
		self.y_pred = self.model.predict(self.x_test)
		print('done')

	def prediction_accuracy(self):
		print('\nCalculating Accuracy of Predictions...',end='')
		self.accuracy = accuracy_score(self.y_test, self.y_pred)
		print('done')

	def confuse(self):
		print('\nCreating confusion matrix...',end='')
		self.cm=confusion_matrix(self.y_test,self.y_pred)
		print('done')

	def print_results(self):
		print(f'=========================================')
		print(f'\nSVM Trained Classifier Accuracy: {self.acc}')
		print(f'Accuracy of Classifier on Test Images: {self.accuracy}\n')
		print(f'=========================================')
		print(f'\nPredicted Values: {self.y_pred}\n')
		print(f'=========================================')
		print(f'\nConfusion Matrix: \n{self.cm}\n')
		print(f'=========================================')

	def plot_results(self):
		plt.matshow(self.cm)
		plt.title('Confusion Matrix for Test Data')
		plt.colorbar()
		plt.ylabel('True label')
		plt.xlabel('Predicted label')
		plt.show()

if __name__=="__main__":
	tlink="poly4_svm.pkl"
	model=SVM(tlink)