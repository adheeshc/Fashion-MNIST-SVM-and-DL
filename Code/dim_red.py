from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA


class myLDA():
	def __init__(self,x_train,y_train,x_test,n):
		self.lda = LDA(n_components=n)
		self.x_train=x_train
		self.y_train=y_train
		self.x_test=x_test
		self.fit()

	def fit(self):
		self.x_train=self.lda.fit_transform(self.x_train,self.y_train)
		self.x_test = self.lda.transform(self.x_test)


class myPCA():
	def __init__(self,x_train,x_test,val):
		self.pca = PCA(val)
		self.x_train=x_train
		self.x_test=x_test
		self.x_train,self.x_test=self.training()
		print(f'Number of components : {pca.n_components_}')

	def fit(self):
		self.x_train=self.pca.fit_transform(self.x_train)
		self.x_test =self.pca.transform(self.x_test)