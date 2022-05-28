from sklearn.naive_bayes import GaussianNB
import time
from preprocessing import preprocess

features_train, labels_train, features_test, labels_test = preprocess()

clf = GaussianNB()
t0 = time()
clf.fit(features_train, labels_train)
print(f"Training time: {round(time()-t0), 3}s")