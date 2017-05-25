import scipy.io
import numpy as np
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import normalize

data = scipy.io.loadmat('data.mat')['data'][0]
bad_patients = [16, 23, 27, 31, 34, 39, 4, 43, 46, 54, 55, 57, 59, 6, 7, 8, 9]


# Command-line arguments 
if len(sys.argv) < 2:
    print "Usage: example_2.py <n> <yes/true/t/y/1 or no/false/f/n/0>"
    sys.exit()


# prune option
prune = True
axis = 0

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

def get_masks(data):
    return [patient[0] for i, patient in enumerate(data) if i not in bad_patients]


def create_dataset(image, mask, n, prune=False):

    subimages = []
    y = []
    for i in range(n, 255-n):
        for j in range(n, 255-n):
            if mask[i,j] != 0:
                
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

if __name__ == "__main__":

    print "making training data"
    masks = get_masks(data)
    t2_normalized = normalize_t2(data)
    # Uncomment for non-normalized t2
    # t2 = get_t2(data)
    n_patients = len(masks)-1


    # Applies gabor filter at different frequencies on patient 0
    # print "filtering images"
    # frequencies = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # test_image = t2_normalized[0]

    # fig, axes = plt.subplots(nrows=1, ncols=9, figsize=(10,3))

    # for frequency, ax in zip(frequencies, axes[0:]):
    #     filt_real, filt_imag = gabor(test_image, frequency=frequency)
    #     #plt.figure()
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
        X_train_single, y_train_single = create_dataset(t2[i], masks[i], n, prune=prune)
        X_train += X_train_single
        y_train += y_train_single


    print "making testing data"


    test_mask = masks[-1]
    test_t2 = t2[-1]
    X_test, y_test = create_dataset(test_t2, test_mask, n, prune = prune)

    #Normalize test data
    X_test = normalize(X_test, axis=axis)

    print "training"
    classifier = RandomForestClassifier()
    classifier.fit(X_train, y_train)

    print "predicting"
    y_pred = classifier.predict(X_test)

    score = roc_auc_score(y_true=y_test, y_score=y_pred)
    print "score", score


