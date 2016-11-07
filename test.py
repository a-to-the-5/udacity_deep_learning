import numpy as np
from sklearn.linear_model import LogisticRegression
from six.moves import cPickle as pickle

save = {}
for s in ['train_dataset', 'train_labels', 'test_dataset', 'test_labels']:
    f = open("%s.pickle" % s, 'rb')
    save[s] = pickle.load(f)
    f.close
train_dataset = save['train_dataset']
train_labels = save['train_labels']
test_dataset = save['test_dataset']
test_labels = save['test_labels']

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

for i in[50, 100, 1000, 5000]: 
  clf = LogisticRegression(C=1e5)
  td = np.reshape(train_dataset[:i,:,:], (i,28*28))
  tr = train_labels[:i]
  clf.fit(td, tr)
  pred = clf.predict(np.reshape(test_dataset, (test_dataset.shape[0], 28*28)))
  print("accuracy for dataset with size %i:" % i, accuracy_score(test_labels, pred))
