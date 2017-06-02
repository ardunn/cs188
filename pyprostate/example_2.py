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
from sklearn.model_selection import LeaveOneOut

data = scipy.io.loadmat('data.mat')['data'][0]

good_patients = [0, 12, 18, 33, 36, 52, 6]


print "Total patients:", len(data)
print "Bad patients:", len(data) - len(good_patients)
n_patients = len(good_patients)
print "Good patients", n_patients

# Command-line arguments 
if len(sys.argv) < 2:
    print "Usage: example_2.py <n> <yes/true/t/y/1 or no/false/f/n/0>"
    sys.exit()

# prune option
prune = True
axis = 0

first_image = False 

n = int(sys.argv[1])
feature_vector_len = (2*n + 1)**2

if sys.argv[2].lower() in ('yes', 'true', 't', 'y', '1'):
    prune = True
if sys.argv[2].lower() in ('no', 'false', 'f', 'n', '0'):
    prune = False

def normalize_t2(data):

    flat_data = []

    for i, patient in enumerate(data):
        if i in good_patients:
            t2 = patient[1]
            flat_data.append(t2.flatten())

    normalized_flat_data = normalize(flat_data)
    normalized_data = [np.reshape(datum, (256, 256)) for datum in normalized_flat_data]
    return normalized_data

def get_t2(data):
    t2_data = []
    for i, patient in enumerate(data):
        if i in good_patients:
            t2 = patient[1]
            t2_data.append(t2)
    return t2_data

def get_filtered_t2(data, frequency):
    filtered_data = []
    for i, patient in enumerate(data):
        if i in good_patients:
            t2 = patient[1]
            filt_real, filt_t2 = gabor(t2, frequency=frequency)
            filtered_data.append(filt_t2)
    return filtered_data

def get_masks(data):
    return [patient[0] for i, patient in enumerate(data) if i in good_patients]

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
                    subimages.append(image[i-n:i+n+1, j-n:j+n+1])
                    # translate (2 -> 1) and 1 -> 0
                    yi = 1 if int(round(mask[i,j]))==2 else 0
                    y.append(yi)
                    row += 1

    X = [image.flatten() for image in subimages]

    return X, y

def create_multiparametric_dataset(image_set, mask, n, prune=False):
    X_tot = []
    _, y = create_dataset(image_set[0], mask, n, prune=prune)

    for image in image_set:
        X, y = create_dataset(image, mask, n, prune=prune)
        X_tot.append(X)

    X_new = []
    for i,_ in enumerate(X_tot[0]):
        Xi_new = []
        for j, _ in enumerate(X_tot):
            Xi_new += X_tot[j][i].tolist()
        X_new.append(Xi_new)

    return X_new, y

def runmodel(model, patient_index=-1, frequency=0.4, quiet=False, silent=False, normalized=False, filtered=False,
             print_example_vector=False):
    """
    Runs a single classifier through sample classifications.
    """

    modelname = model.__class__.__name__

    if not quiet:
        print "{model}: patient {patient}/{n_patients}: making training data".format(model=modelname, n_patients=n_patients-1, patient=patient_index)

    masks = get_masks(data)

    t2_filtered = get_filtered_t2(data, frequency=frequency)

    if normalized:
        t2 = normalize_t2(data)
    else:
        t2 = get_t2(data)

    if patient_index == -1:
        patient_index = n_patients - 1

    X_train = []
    y_train = []

    # Create training data

    for i in range(n_patients):

        if i == patient_index:
            continue

        else:

            if filtered:
                image_set = [t2[i], t2_filtered[i]]
                X_train_single, y_train_single = create_multiparametric_dataset(image_set, masks[i], n, prune=prune)
            else:
                X_train_single, y_train_single = create_dataset(t2[i], masks[i], n, prune=prune)

            X_train += X_train_single
            y_train += y_train_single

    if print_example_vector:
        ex_vec = X_train[0]
        print ex_vec

    # Create testing data

    if not quiet:
        print "{model}: patient {patient}/{n_patients}: making testing data".format(model=modelname, n_patients=n_patients-1, patient=patient_index)

    test_mask = masks[patient_index]
    test_t2 = t2[patient_index]
    test_t2_filtered = t2_filtered[patient_index]
    if filtered:
        image_set = [test_t2, test_t2_filtered]
        X_test, y_test = create_multiparametric_dataset(image_set, test_mask, n, prune = prune)
    else:
        X_test, y_test = create_dataset(test_t2, test_mask, n, prune=prune)


    # Train, predict, and score

    if not quiet:
        print "{model}: patient {patient}/{n_patients}: training".format(model=modelname, n_patients=n_patients-1, patient=patient_index)
    model.fit(X_train, y_train)

    if not quiet:
        print "{model}: patient {patient}/{n_patients}: testing".format(model=modelname, n_patients=n_patients-1, patient=patient_index)
    y_pred = model.predict(X_test)

    score = roc_auc_score(y_true=y_test, y_score=y_pred)

    if not silent:
        print "{model}: patient {patient}/{n_patients}: score {score}".format(model=modelname, n_patients=n_patients-1, score=score, patient=patient_index)

    return score

def crossvalidate(*args, **kwargs):

    scores = []
    j = 0
    for i, _ in enumerate(data):
        if i in good_patients:

            print "real patient index:", i
            kwargs['patient_index'] = j
            score = runmodel(*args, **kwargs)
            scores.append(score)
            j += 1

    print "Overall cross validated score", np.mean(scores)
    return np.mean(scores)



if __name__ == "__main__":

    model = RandomForestClassifier(n_estimators=10)
    # runmodel(model, filtered=True, frequency=0.4)
    # runmodel(model)
    crossvalidate(model, filtered=True, print_example_vector=True)

    # For examining effect of gabor frequency filtering
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


    





