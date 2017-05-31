import scipy.io
import numpy as np
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import normalize
from skimage.filters import gabor
from skimage import data, io
from matplotlib import pyplot as plt

data = scipy.io.loadmat('data.mat')['data'][0]
bad_patients = [16, 23, 27, 31, 34, 39, 4, 43, 46, 54, 55, 57, 59, 6, 7, 8, 9]
bad_patients2 = [1, 10, 13, 14, 15, 17, 19, 2, 20, 22, 24, 25, 26, 29, 3, 32, 37, 38, 40, 41, 42, 44, 45, 47, 48, 49,
                 5, 51, 52, 53, 58, 60, 61]
bad_patients += bad_patients2


# Command-line arguments 
if len(sys.argv) < 2:
    print "Usage: example_2.py <n> <yes/true/t/y/1 or no/false/f/n/0>"
    sys.exit()


# prune option
prune = True
axis = 0

first_image = False 

n = int(sys.argv[1])

if sys.argv[2].lower() in ('yes', 'true', 't', 'y', '1'):
    prune = True
if sys.argv[2].lower() in ('no', 'false', 'f', 'n', '0'):
    prune = False

def normalize_t2(data):

    flat_data = []

    for i, patient in enumerate(data):
        if i not in bad_patients:
            t2 = patient[1]
            flat_data.append(t2.flatten())

    normalized_flat_data = normalize(flat_data)
    normalized_data = [np.reshape(datum, (256, 256)) for datum in normalized_flat_data]
    return normalized_data

def get_t2(data):
    t2_data = []
    for i, patient in enumerate(data):
        if i not in bad_patients:
            t2 = patient[1]
            t2_data.append(t2)
    return t2_data

def get_masks(data):
    return [patient[0] for i, patient in enumerate(data) if i not in bad_patients]

def create_dataset(image, mask, n, prune=False):
    row = 1
    subimages = []
    y = []
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
                    subimage = image[i-n:i+n+1, j-n:j+n+1]
                    if i == 105 and j == 105 and first_image:
                        print "saving subimage at (105,105)"
                        np.savetxt('subimage_105-105.csv', subimage, delimiter=',')
                        print row
                    subimages.append(image[i-n:i+n+1, j-n:j+n+1])
                    # translate (2 -> 1) and 1 -> 0
                    yi = 1 if int(round(mask[i,j]))==2 else 0
                    y.append(yi)
                    row += 1

    X = [image.flatten() for image in subimages]

    return X, y

if __name__ == "__main__":

    print "making training data"
    masks = get_masks(data)
    # t2_normalized = normalize_t2(data)
    # Uncomment for non-normalized t2
    t2 = get_t2(data)
    n_patients = len(masks)-1

    print "saving patient 0 normalized data array"
    np.savetxt('normalized_array.csv', t2[0], delimiter=',')

    print "saving patient 0 mask data array"
    np.savetxt('mask_array.csv', masks [0], delimiter=',')

    # Applies gabor filter at different frequencies on patient 0
    # print "filtering images"
    # frequencies = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # test_image = t2_normalized[0]
    #
    # fig, axes = plt.subplots(nrows=1, ncols=9, figsize=(10,3))

    # for frequency, ax in zip(frequencies, axes[0:]):
    #     filt_real, filt_imag = gabor(test_image, frequency=frequency)
    #     image_name = 'patient{}_{}'.format(0, frequency)
    #     ax.imshow(filt_real)
    #     ax.axis('off')
    #     ax.set_title(frequency, fontsize=12)
    #     #plt.savefig('{}.png'.format(image_name))
    #     #plt.close()
    
    # plt.show()
    
    X_train = []
    y_train = []

    for i in range(n_patients):
        first_image = True if i==0 else False
        X_train_single, y_train_single = create_dataset(t2[i], masks[i], n, prune=prune)
        X_train += X_train_single
        y_train += y_train_single
        if i == 0: 
            print "Saving y training array"
            np.savetxt('training_y_array.csv', y_train_single, delimiter=',')
            print "Saving X training array"
            np.savetxt('training_X_array.csv', X_train_single, delimiter=',')

    print "making testing data"

    test_mask = masks[-1]
    test_t2 = t2[-1]
    X_test, y_test = create_dataset(test_t2, test_mask, n, prune = prune)

    print "training"
    classifier = RandomForestClassifier(n_estimators=2)
    # classifier = MLPClassifier()
    classifier.fit(X_train, y_train)


    print "predicting"
    y_pred = classifier.predict(X_test)

    score = roc_auc_score(y_true=y_test, y_score=y_pred)
    print "Saving y test"
    np.savetxt('y_test_array.csv', y_test, delimiter=',')
    print "Saving y pred"
    np.savetxt('y_pred_array.csv', y_pred, delimiter=',')
    print "score", score


