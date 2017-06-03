from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.metrics import binary_accuracy
from sklearn.metrics import roc_auc_score
import numpy as np
import scipy.io
import sys
import tensorflow as tf

def create_dataset(image, mask, n, prune=False):
    row = 1
    X = []
    y = []
    acpx = []
    for i in range(n, 255-n):
        for j in range(n, 255-n):
            if mask[i,j]!=0:
                # numpy submatrices have (first index starting at 0):(last index starting from 1)

                store = True
                if prune:
                    for v in mask[i-n:i+n+1, j-n:j+n+1]:
                        if 0 in v:
                            store = False
                            break

                if store:
                    acpx.append((i, j))
                    subimage = image[i-n:i+n+1, j-n:j+n+1]
                    X.append(subimage)
                    # translate (2 -> 1) and 1 -> 0
                    yi = 1 if int(round(mask[i,j]))==2 else 0
                    y.append(yi)
                    row += 1
    return X, y


def roc_wrapper(Y_true, Y_pred):
    with sess.as_default():
        print Y_true.eval()
        print Y_pred.eval()


if __name__ == "__main__":
    data = scipy.io.loadmat('data.mat')['data'][0]
    good_patients = [0, 12, 18, 33, 36, 3, 24, 45, 48, 52, 58, 61]

    print "Total patients:", len(data)
    print "Bad patients:", len(data) - len(good_patients)
    n_patients = len(good_patients)
    print "Good patients", n_patients, "\n"

    # Command-line arguments
    if len(sys.argv) < 2:
        print "Usage: example_2.py <n> <yes/true/t/y/1 or no/false/f/n/0>"
        sys.exit()

    # prune option
    prune = True
    n = int(sys.argv[1])
    subimage_size = 2*n + 1

    if sys.argv[2].lower() in ('yes', 'true', 't', 'y', '1'):
        prune = True
    if sys.argv[2].lower() in ('no', 'false', 'f', 'n', '0'):
        prune = False

    masks = np.asarray([patient[0] for i, patient in enumerate(data) if i in good_patients])
    t2 = np.asarray([patient[1] for i, patient in enumerate(data) if i in good_patients])

    sess = tf.Session()

    test_patient = -1

    X_train = []
    y_train = []

    for i in range(n_patients):
        if i != test_patient:

            X_train_single, y_train_single = create_dataset(t2[i], masks[i], n, prune=prune)
            X_train += X_train_single
            y_train += y_train_single


    X_test, y_test = create_dataset(t2[test_patient], masks[test_patient], n, prune=prune)

    X_test = np.asarray(X_test)
    X_train = np.asarray(X_train)

    X_test = X_test.reshape(X_test.shape[0], subimage_size, subimage_size, 1).astype('float32')
    X_train = X_train.reshape(X_train.shape[0], subimage_size, subimage_size, 1).astype('float32')

    y_test = np.asarray(y_test)
    y_train = np.asarray(y_train)

    Y_test = np_utils.to_categorical(y_test, 2)
    Y_train = np_utils.to_categorical(y_train, 2)

    model = Sequential()
    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(subimage_size, subimage_size, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[binary_accuracy])
    model.fit(X_train, Y_train, verbose=2, epochs=10)

    Y_pred = model.predict(X_test)
    score = roc_auc_score(y_true=Y_test, y_score=Y_pred)
    print score


