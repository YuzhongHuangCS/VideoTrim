import pickle
import pdb
import numpy as np
import sklearn
import sklearn.svm
import sklearn.dummy

def flatten_nested_list(nested_list):
    return [item for sublist in nested_list for item in sublist]

with open('train_emb.pickle', 'rb') as fin:
    [train_emb_ary, train_label_ary] = pickle.load(fin)

with open('valid_emb.pickle', 'rb') as fin:
    [valid_emb_ary, valid_label_ary] = pickle.load(fin)

with open('test_emb.pickle', 'rb') as fin:
    [test_emb_ary, test_label_ary] = pickle.load(fin)

train_emb_ary = np.asarray(flatten_nested_list(train_emb_ary))
valid_emb_ary = np.asarray(flatten_nested_list(valid_emb_ary))
test_emb_ary = np.asarray(flatten_nested_list(test_emb_ary))
train_label_ary = np.asarray(flatten_nested_list(train_label_ary))
valid_label_ary = np.asarray(flatten_nested_list(valid_label_ary))
test_label_ary = np.asarray(flatten_nested_list(test_label_ary))

from sklearn.decomposition import PCA
print('Before pca')
pca = PCA(n_components=2, svd_solver='randomized')
train_emb_2d = pca.fit_transform(train_emb_ary)
print('After pca')

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets

X = train_emb_2d
                      # avoid this ugly slicing by using a two-dim dataset
y = train_label_ary

h = .02  # step size in the mesh

# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors
C = 1.0  # SVM regularization parameter
svc = svm.SVC(kernel='linear', C=C).fit(X, y)
rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X, y)
poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X, y)
lin_svc = svm.LinearSVC(C=C).fit(X, y)

# create a mesh to plot in
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# title for the plots
titles = ['SVC with linear kernel',
          'LinearSVC (linear kernel)',
          'SVC with RBF kernel',
          'SVC with polynomial (degree 3) kernel']

#for i, clf in enumerate((lin_svc,)):
for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):
    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    plt.subplot(2, 2, i + 1)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title(titles[i])

plt.savefig('full_svm.png')
print('123')