import pickle
import pdb
import numpy as np
import sklearn
import sklearn.svm
import sklearn.dummy

def flatten_nested_list(nested_list):
    return [item for sublist in nested_list for item in sublist]

with open('baseline_extract.pickle', 'rb') as fin:
    [train_diff_framewise, train_diff_mean, train_Y_ary,valid_diff_framewise, valid_diff_mean, valid_Y_ary, test_diff_framewise, test_diff_mean, test_Y_ary] = pickle.load(fin)

train_diff_framewise = np.asarray(train_diff_framewise)
train_diff_mean = np.asarray(train_diff_mean)
train_Y_ary = np.asarray(flatten_nested_list(train_Y_ary))
valid_diff_framewise = np.asarray(valid_diff_framewise)
valid_diff_mean = np.asarray(valid_diff_mean)
valid_Y_ary = np.asarray(flatten_nested_list(valid_Y_ary))
test_diff_framewise = np.asarray(test_diff_framewise)
test_diff_mean = np.asarray(test_diff_mean)
test_Y_ary = np.asarray(flatten_nested_list(test_Y_ary))

dummy = sklearn.dummy.DummyClassifier()
dummy.fit(train_diff_framewise, train_Y_ary)
print('dummy framewise', dummy.score(train_diff_framewise, train_Y_ary))
print('dummy framewise', dummy.score(valid_diff_framewise, valid_Y_ary))
print('dummy framewise', dummy.score(test_diff_framewise, test_Y_ary))

for c in [0.1, 0.2, 0.25, 0.5, 0.8, 1, 1.25, 1.5, 2, 2.5, 2, 2.5, 4, 5, 8, 10]:
    print('c=', c)
    '''
    svm = sklearn.svm.SVC(C=c)
    svm.fit(train_diff_framewise, train_Y_ary)
    print('svm framewise, C=', c, svm.score(train_diff_framewise, train_Y_ary))
    print('svm framewise, C=', c, svm.score(valid_diff_framewise, valid_Y_ary))
    print('svm framewise, C=', c, svm.score(test_diff_framewise, test_Y_ary))
    test_pred = svm.predict(test_diff_framewise)
    print(sklearn.metrics.confusion_matrix(test_Y_ary, test_pred))


    svm = sklearn.svm.SVC(C=c)
    svm.fit(train_diff_mean, train_Y_ary)
    print('svm mean, C=', c, svm.score(train_diff_mean, train_Y_ary))
    print('svm mean, C=', c, svm.score(valid_diff_mean, valid_Y_ary))
    print('svm mean, C=', c, svm.score(test_diff_mean, test_Y_ary))
    test_pred = svm.predict(test_diff_mean)
    print(sklearn.metrics.confusion_matrix(test_Y_ary, test_pred))
    '''
    svm = sklearn.svm.LinearSVC(C=c)
    svm.fit(train_diff_framewise, train_Y_ary)
    print('LinearSVC framewise, C=', c, svm.score(train_diff_framewise, train_Y_ary))
    print('LinearSVC framewise, C=', c, svm.score(valid_diff_framewise, valid_Y_ary))
    print('LinearSVC framewise, C=', c, svm.score(test_diff_framewise, test_Y_ary))
    test_pred = svm.predict(test_diff_framewise)
    print(sklearn.metrics.confusion_matrix(test_Y_ary, test_pred))

    #svm = sklearn.svm.LinearSVC(C=c)
    #svm.fit(train_diff_mean, train_Y_ary)
    #print('LinearSVC mean, C=', c, svm.score(train_diff_mean, train_Y_ary))
    #print('LinearSVC mean, C=', c, svm.score(valid_diff_mean, valid_Y_ary))
    #print('LinearSVC mean, C=', c, svm.score(test_diff_mean, test_Y_ary))
    #test_pred = svm.predict(test_diff_mean)
    #print(sklearn.metrics.confusion_matrix(test_Y_ary, test_pred))

#train_diff_framewise = flatten_nested_list(train_diff_framewise)
pdb.set_trace()
print('123')