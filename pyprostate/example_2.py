import scipy.io
import numpy as np
import sys
from sklearn.ensemble import RandomForestClassifier,  GradientBoostingClassifier, AdaBoostClassifier, \
    ExtraTreesClassifier, BaggingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import normalize
from skimage.filters import gabor
from skimage import data, io
from matplotlib import pyplot as plt
from sklearn.model_selection import LeaveOneOut
import scipy.misc

#todo: document, refactor, organize

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

def get_filtered_t2(data, frequency, sigma_x=None, sigma_y=None):
    filtered_data = []
    norm_data = normalize_t2(data)
    for patient_im in norm_data:
            filt_real, filt_t2 = gabor(patient_im, frequency=frequency, sigma_x=sigma_x, sigma_y=sigma_y)
            filtered_data.append(filt_real)
    return filtered_data

def get_masks(data):
    return [patient[0] for i, patient in enumerate(data) if i in good_patients]

def create_dataset(image, mask, n, prune=False, save_pxs=False):
    row = 1
    subimages = []
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
                    subimages.append(subimage)
                    # translate (2 -> 1) and 1 -> 0
                    yi = 1 if int(round(mask[i,j]))==2 else 0
                    y.append(yi)
                    row += 1

    if save_pxs:
        active_pixels.append(acpx)
    X = [image.flatten() for image in subimages]
    return X, y

def create_multiparametric_dataset(image_set, mask, n, prune=False, save_pxs=False):
    X_tot = []
    _, y = create_dataset(image_set[0], mask, n, prune=prune, save_pxs=save_pxs)

    for image in image_set:
        X, _ = create_dataset(image, mask, n, prune=prune)
        X_tot.append(X)

    X_new = []
    for i,_ in enumerate(X_tot[0]):
        Xi_new = []
        for j, _ in enumerate(X_tot):
            Xi_new += X_tot[j][i].tolist()
        X_new.append(Xi_new)

    return X_new, y

def reconstruct(mask, y_pred, pixels):

    # change mask so that cancer prediction is gone (ie prevent cheating)
    for pi in range(256):
        for pj in range(256):
            if int(round(mask[pi, pj])) == 2:
                mask[pi, pj] = 1

    # insert the actual prediction
    for p, pixel in enumerate(pixels):
        pixel_x = pixel[0]
        pixel_y = pixel[1]
        # set the new mask pixel to the predicted score, moving 0 -> 1 (prostate, not cancer)
        #  and 1 -> 2 (prostate, cancer)
        mask[pixel_x, pixel_y] = y_pred[p] + 1

    return mask

def runmodel(model, patient_index=-1, frequency=0.4, quiet=False, silent=False, normalized=False, filtered=False,
             print_example_vector=False, save_reconstruction=False):
    """
    Runs a single classifier through sample classifications.
    """

    modelname = model.__class__.__name__

    if silent:
        quiet = True
    if not quiet:
        print "{model}: patient {patient}/{n_patients}: making training data".format(model=modelname, n_patients=n_patients, patient=patient_index+1)

    masks = get_masks(data)

    if normalized:
        t2 = normalize_t2(data)
    else:
        t2 = get_t2(data)

    if patient_index == -1:
        patient_index = n_patients - 1

    t2_filtered = get_filtered_t2(data, frequency=frequency, sigma_x=2.0, sigma_y=2.0)
    # t2_filtered2 = get_filtered_t2(data, frequency=frequency, sigma_y=2.0, sigma_x=2.0)

    image_set_total = [t2]
    image_set_total.append(t2_filtered)

    X_train = []
    y_train = []

    # Create training data

    for i in range(n_patients):

        if i == patient_index:
            continue

        else:

            if filtered:
                image_set = [imset[i] for imset in image_set_total]
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
        print "{model}: patient {patient}/{n_patients}: making testing data".format(model=modelname, n_patients=n_patients, patient=patient_index+1)

    test_mask = masks[patient_index]
    test_t2 = t2[patient_index]
    if filtered:
        image_set = [imset[patient_index] for imset in image_set_total]
        X_test, y_test = create_multiparametric_dataset(image_set, test_mask, n, prune=prune, save_pxs=True)
    else:
        X_test, y_test = create_dataset(test_t2, test_mask, n, prune=prune, save_pxs=True)

    # Train, predict, and score

    if not quiet:
        print "{model}: patient {patient}/{n_patients}: training using {n_pts} points".format(model=modelname, n_patients=n_patients, patient=patient_index+1, n_pts=len(y_train))
    model.fit(X_train, y_train)

    if not quiet:
        print "{model}: patient {patient}/{n_patients}: testing using {n_pts} points".format(model=modelname, n_patients=n_patients, patient=patient_index+1, n_pts=len(y_test))
    y_pred = model.predict(X_test)

    score = roc_auc_score(y_true=y_test, y_score=y_pred)

    if not silent:
        print "{model}: patient {patient}/{n_patients}: score {score}".format(model=modelname, n_patients=n_patients, score=score, patient=patient_index+1)

    if save_reconstruction:
        if not silent:
            print "{model}: patient {patient}/{n_patients}: saving reconstructed mask".format(model=modelname, n_patients=n_patients, patient=patient_index+1)
        reconstruction = reconstruct(test_mask, y_pred, active_pixels[patient_index])
    else:
        reconstruction = None

    return score, reconstruction

def crossvalidate(*args, **kwargs):

    scores = []
    j = 0
    for i, _ in enumerate(data):
        if i in good_patients:

            if 'silent' in kwargs:
                if kwargs['silent']:
                    pass
                else:
                    print "real patient index:", i
            else:
                print "real patient index:", i

            kwargs['patient_index'] = j
            score, reconstruction = runmodel(*args, **kwargs)
            scores.append(score)

            if 'save_reconstruction' in kwargs:
                if kwargs['save_reconstruction']:
                    scipy.misc.imsave("patient_{}_reconstruction.png".format(i), reconstruction)
            j += 1

    cvmodel = args[0].__class__.__name__
    print "{} overall cross validated score {}".format(cvmodel, np.mean(scores))
    return np.mean(scores)


if __name__ == "__main__":

    data = scipy.io.loadmat('data.mat')['data'][0]

    # unscorable patients (ie all cancer)
    unscorable_patients = [27]

    # new patients 3, 24, 45, 48
    # good_patients = [0, 12, 18, 33, 36, 52]
    # possible good new patients: 58, 61
    good_patients = [0, 12, 18, 33, 36, 3, 24, 45, 48, 52, 58, 61]
    # good_patients = [0, 12, 33, 36, 52]

    print "Total patients:", len(data)
    print "Bad patients:", len(data) - len(good_patients)
    n_patients = len(good_patients)
    print "Good patients", n_patients, "\n"

    # Command-line arguments
    if len(sys.argv) < 2:
        print "Usage: example_2.py <n> <yes/true/t/y/1 or no/false/f/n/0>"
        sys.exit()

    active_pixels = []

    # prune option
    prune = True
    axis = 0

    n = int(sys.argv[1])
    feature_vector_len = (2 * n + 1) ** 2

    if sys.argv[2].lower() in ('yes', 'true', 't', 'y', '1'):
        prune = True
    if sys.argv[2].lower() in ('no', 'false', 'f', 'n', '0'):
        prune = False

    model = RandomForestClassifier(n_estimators=10)
    model2 = GradientBoostingClassifier()
    model3 = AdaBoostClassifier()
    model4 = ExtraTreesClassifier()
    model5 = BaggingClassifier()
    model6 = SVC()

    # for f in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    #
    #     print "running with frequency of", f
    #     crossvalidate(model2, filtered=True, frequency=f, silent=True)
    #

    # for k in range(28, len(data)):
    #     if k not in good_patients:
    #         good_patients.append(k)
    #
    #         print "Adding candidate patient", k, "to good_patients"
    #         crossvalidate(model2, filtered=False, quiet=False, frequency=0.1, silent=True)
    #
    #         good_patients.remove(k)

    crossvalidate(model2, filtered=True, quiet=False, frequency=0.1, normalized=True, silent=False, save_reconstruction=True)

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


    





