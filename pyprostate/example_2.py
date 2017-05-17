import scipy.io
import numpy as np
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

data = scipy.io.loadmat('data.mat')['data'][0]

#patient = data[0]
#mask = patient[0]
#t2 = patient[1]

# Testing Data
patient2 = data[-1]
mask2 = patient2[0]
t2_2 = patient2[1]

# Command-line arguments 
if len(sys.argv) < 2:
    print "Usage: example_2.py <n> <yes/true/t/y/1 or no/false/f/n/0>"
    sys.exit()

n = int(sys.argv[1])

if sys.argv[2].lower() in ('yes', 'true', 't', 'y', '1'): 
    prune = True
if sys.argv[2].lower() in ('no', 'false', 'f', 'n', '0'):
    prune = False

def create_dataset(image, mask, n, prune=False):

    subimages = []
    y = []
    for i in range(n, 255-n):
        for j in range(n, 255-n):
            if mask[i,j] != 0:
                `
                # numpy submatrices have (first index starting at 0):(last index starting from 1)

                store = True
                if prune:
                    for v in mask[i-n:i+n+1, j-n:j+n+1]:
                        if 0 in v:
                            store = False
                            break

                if store:
                    subimages.append(image[i-n:i+n+1, j-n:j+n+1])

                    # translate (2 -> 1) and 1 -> 0
                    yi = 1 if int(round(mask[i,j]))==2 else 0
                    y.append(yi)

    X = [image.flatten() for image in subimages]
    return X, y


print "making training data"
X_train = []
y_train = []

for image in data[0:-1]:
    mask_single = image[0]
    t2_single = image[1]
    X_train_single, y_train_single = create_dataset(t2_single, mask_single, 3, prune=True)
    X_train += X_train_single
    y_train += y_train_single

print "making testing data"
X_test, y_test = create_dataset(t2_2, mask2, 3, prune = True)

print "training"
classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)

print "predicting"
y_pred = classifier.predict(X_test)

score = roc_auc_score(y_true=y_test, y_score=y_pred)
print "score", score


