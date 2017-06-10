"""
This was our first attempt at using machine learning to classify the images. Don't pay much attention to this code, 
the data conversion from image to feature vector is not done right and our ROC AUC scores are very low. 
"""

import scipy.io
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot as plt

# X is the collection of all feature vectors for all data points
# y is the collection of all output vectors

def create_subimages(single_image, n):
    subimages = []
    for i in range(0,256, n):
        for j in range(0, 256, n):
            subimages.append(single_image[i:i+n,j:j+n])
    return subimages

def create_X(single_image, n):
    subimages = create_subimages(single_image, n)
    return [image.flatten() for image in subimages]

def create_y(mask, n):

    # todo: right now this just bases the output vector on whether a single 1 or 2 is present in the nxn subimage
    # todo: NOT whether the point is actually cancerous or not. we should make this a probability maybe based on the
    # todo: number of 2s and 1s in the image.
    f_vectors = create_X(mask, n)

    y = [1 if 1 in vector or 2 in vector else 0 for vector in f_vectors]
    return y


def example_run(n, data):

    # data is a 1x62 array
    # data[0] is a 62x1
    # data[0][0] is a structure of a single patients images
    # data[0][0][0] a single image as an array

    # transform training data
    t2_train = data[0][0][1]
    mask_train = data[0][0][0]
    X_train = create_X(t2_train, n)
    y_train = create_y(mask_train, n)

    # train the model
    classifier = RandomForestClassifier()
    classifier.fit(X_train, y_train)

    # transform testing data
    t2_test = data[0][1][1]
    mask_test = data[0][1][0]
    X_test = create_X(t2_test, n)
    y_test = create_y(mask_test, n)

    # test the model
    y_pred = classifier.predict(X_test)
    score = roc_auc_score(y_true=y_test, y_score=y_pred)

    # save the graphic results
    image_pred = np.reshape(y_pred, (256/n, 256/n))
    image_true = np.reshape(y_test, (256/n, 256/n))

    # plt.figure(n)
    # plt.imshow(image_pred, interpolation='nearest')
    # plt.savefig('{}_predicted.png'.format(n))
    # plt.close()

    # plt.figure(n+1)
    # plt.imshow(image_true, interpolation='nearest')
    # plt.savefig('{}_true.png'.format(n))
    # plt.close()

    return score


if __name__ == "__main__":
    mat = scipy.io.loadmat('data.mat')
    # np.set_printoptions(threshold=np.inf)

    data = mat['data']

    n_sizes = [2,4,8,16]
    scores = [example_run(n, data) for n in n_sizes]

    plt.figure(1)
    plt.plot(n_sizes, scores)
    plt.xlabel("n, Size of subimage is n x n")
    plt.ylabel("ROC AUC Curve (how good the model is)")
    plt.show()









