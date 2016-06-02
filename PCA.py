##################################################################################
#About
##################################################################################
#script: PCA.py
#purpose: Show an an example of principle component analysis (PCA) using movie 
#         reccommender data
#data source: https://movielens.org; https://github.com/warriorkitty/orientlens/tree/master/movielens
#code source: http://sebastianraschka.com/Articles/2015_pca_in_3_steps.html

##################################################################################
#Imports
##################################################################################

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from matplotlib import*
import matplotlib.pyplot as plt
from matplotlib.cm import register_cmap
from scipy import stats
from wpca import PCA
from sklearn.decomposition import PCA as sklearnPCA
import seaborn


##################################################################################
#Data
##################################################################################
#Load movie names and movie ratings
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')
ratings.drop(['timestamp'], axis=1, inplace=True)

def replace_name(x):
	return movies[movies['movieId']==x].title.values[0]
ratings.movieId = ratings.movieId.map(replace_name)

M = ratings.pivot_table(index=['userId'], columns=['movieId'], values='rating')
m = M.shape
print m

df1 = M.replace(np.nan, 0, regex=True)
X_std = StandardScaler().fit_transform(df1)


##################################################################################
#PCA Analysis
##################################################################################
#Create a covariance matrix
mean_vec = np.mean(X_std, axis=0)
cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0]-1)
print('Covariance matrix \n%s' %cov_mat)

#Create the same covariance matrix with 1 line of code
print('NumPy covariance matrix: \n%s' %np.cov(X_std.T))

#Perform eigendecomposition on covariance matrix
cov_mat = np.cov(X_std.T)
eig_vals, eig_vecs = np.linalg.eig(cov_mat)
print('Eigenvectors \n%s' %eig_vecs)
print('\nEigenvalues \n%s' %eig_vals)

# Visually confirm that the list is correctly sorted by decreasing eigenvalues
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
print('Eigenvalues in descending order:')
for i in eig_pairs:
    print(i[0])

#Variable loadings
pca = PCA(n_components=2)
pca.fit_transform(df1)
i = np.identity(df1.shape[1]) 
coef = pca.transform(i)
loadings = pd.DataFrame(coef, columns=['PC-1', 'PC-2'], index=M.columns)
print loadings.head(10)

#print the eigenvalues for the first two principle components
print pca.explained_variance_ratio_ 

#Explained variance
pca = PCA().fit(X_std)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()


