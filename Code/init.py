import numpy as np
import sys
sys.path.insert(1, '../fashion_mnist/utils')
import mnist_reader
import matplotlib.pyplot as plt
import math
import pickle as pkl
from sklearn import svm
import os

class initializeMNIST():
	def __init__(self):
		self.x_train,self.y_train,self.x_test,self.y_test=self.load_data()
		#self.mean_list,self.cov_list=self.load_model()

	def load_data(self):
		x_train,y_train=mnist_reader.load_mnist('../fashion_mnist/data/fashion',kind='train')
		x_test,y_test=mnist_reader.load_mnist('../fashion_mnist/data/fashion',kind='t10k')
		return x_train,y_train,x_test,y_test

	def load_model(self):
		if os.path.isfile("../Datadumps/"+"mean_list.pkl"):
			print("Load existing pickle dumps...")
			mean_list=pkl.load(open("../Datadumps/mean_list.pkl","rb"))
		if os.path.isfile("../Datadumps/"+"cov_list.pkl"):
			cov_list=pkl.load(open("../Datadumps/cov_list.pkl","rb"))
		else:
			print("Creating new pickle dumps...")
			mean_list,cov_list=self.train_dict()
		return mean_list,cov_list

	def print_data(self):
		print(f'x train shape: {self.x_train.shape}')
		print(f'y train shape: {self.y_train.shape} with labels {set(self.y_train)}')
		print(f'x test shape: {self.x_test.shape}')
		print(f'y test shape: {self.y_test.shape} with labels {set(self.y_test)}')
		print(f'number of x_train: {self.x_train.shape[0]} ')
		print(f'number of x_test: {self.x_test.shape[0]} ')

	def visualize_data(self):
		index = np.random.randint(self.x_train.shape[0],size=9)
		for i in range(0,9):
			plt.subplot(3,3,i+1)
			plt.imshow(self.x_train[index[i]].reshape(28,28))
			plt.title(f'Index: {index[i]}, Label: {self.y_train[index[i]]} ')
			plt.colorbar()
			plt.grid(False)
			plt.tight_layout()
		plt.show()

	def train_dict(self):
		list0=[];list1=[];list2=[];list3=[];list4=[];list5=[];list6=[];list7=[];list8=[];list9=[]
		for idx,label in enumerate(self.y_train):
			if label==0:list0.append(self.x_train[idx])
			if label==1:list1.append(self.x_train[idx])
			if label==2:list2.append(self.x_train[idx])
			if label==3:list3.append(self.x_train[idx])
			if label==4:list4.append(self.x_train[idx])
			if label==5:list5.append(self.x_train[idx])
			if label==6:list6.append(self.x_train[idx])
			if label==7:list7.append(self.x_train[idx])
			if label==8:list8.append(self.x_train[idx])
			if label==9:list9.append(self.x_train[idx])
		list0=np.array(list0);mlist0=np.mean(list0,axis=0);clist0=np.cov(list0.T)
		list1=np.array(list1);mlist1=np.mean(list1,axis=0);clist1=np.cov(list1.T)
		list2=np.array(list2);mlist2=np.mean(list2,axis=0);clist2=np.cov(list2.T)
		list3=np.array(list3);mlist3=np.mean(list3,axis=0);clist3=np.cov(list3.T)
		list4=np.array(list4);mlist4=np.mean(list4,axis=0);clist4=np.cov(list4.T)
		list5=np.array(list5);mlist5=np.mean(list5,axis=0);clist5=np.cov(list5.T)
		list6=np.array(list6);mlist6=np.mean(list6,axis=0);clist6=np.cov(list6.T)
		list7=np.array(list7);mlist7=np.mean(list7,axis=0);clist7=np.cov(list7.T)
		list8=np.array(list8);mlist8=np.mean(list8,axis=0);clist8=np.cov(list8.T)
		list9=np.array(list9);mlist9=np.mean(list9,axis=0);clist9=np.cov(list9.T)
		cov_list=[]
		mean_list=np.vstack((mlist0,mlist1,mlist2,mlist3,mlist4,mlist5,mlist6,mlist7,mlist8,mlist9))
		cov_list.extend((clist0,clist1,clist2,clist3,clist4,clist5,clist6,clist7,clist8,clist9))
		cov_list=np.array(cov_list)
		pkl.dump(mean_list,open("../Datadumps/mean_list.pkl","wb"))
		pkl.dump(cov_list,open("../Datadumps/cov_list.pkl","wb"))
		return mean_list,cov_list
		
	def normalize_data(self):
		self.x_train = self.x_train/255
		self.x_test = self.x_test/255

	def reshape_data(self):
		im_rows=28
		im_cols=28
		im_shape=(im_rows,im_cols,1)
		self.x_train=self.x_train.reshape(self.x_train.shape[0],*im_shape)
		self.x_test=self.x_test.reshape(self.x_test.shape[0],*im_shape)


if __name__=="__main__":
	init=initializeMNIST()
	x=init.x_train
	print(x[0].shape)
