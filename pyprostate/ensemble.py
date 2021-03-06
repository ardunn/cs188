"""
This file contains the core code for running ensemble methods on the image data, and was the main workhorse of our 
project. Contained in this file are the functions for:

    1. Acquiring and refining data
    2. Forming sets of vectorized input and output for training.
    3. Testing and scoring
    4. Cross validation
    5. Reconstruction of images from machine learning predictions. 
    
Brief docstrings are given in each function to explain its purpose. Block comments clarify specific lines.

To call the code in this file, enter in the command line:
    "python ensemble.py 3 y"
    
Where 3 and y are command line arguments. 

The first command line argument is for n, the number of pixels on each side of a data point pixel to use in machine
learning. Typically, higher n are more accurate (because it includes more data) but take longer for training and testing
because the dimensionality of the problem increases. 
 
The second command line argument is whether to "prune" the data or not, which removes pixels from training and testing
if the pixel (or surrounding pixels used as features) are outside the mask area. For example, if a pixel inside the 
prostate area has an associated feature vector with pixels outside the prostate area, it will not be included. 
"""
import scipy.io
import numpy as np
import sys
from sklearn.ensemble import RandomForestClassifier,  GradientBoostingClassifier, AdaBoostClassifier, \
    ExtraTreesClassifier, BaggingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import normalize
from skimage.filters import gabor
from skimage import data
import scipy.misc

# Load the data from the data.mat file given by Dr. Scalzo
data = scipy.io.loadmat('data.mat')['data'][0]

# Patients which cannot be scored due to them being all cancer
unscorable_patients = [27]

# Patients kept for analysis based on Dr. Scalzo's recommendations
good_patients = [0, 12, 18, 33, 36, 3, 24, 45, 48, 52, 58, 61]

# Print run information
print "Total patients:", len(data)
print "Bad patients:", len(data) - len(good_patients)
n_patients = len(good_patients)
print "Good patients", n_patients, "\n"

# Command-line arguments
if len(sys.argv) < 2:
    print "Usage: ensemble.py <n> <yes/true/t/y/1 or no/false/f/n/0>"
    sys.exit()

active_pixels = []

# Command line arguments
prune = True
axis = 0

n = int(sys.argv[1])
feature_vector_len = (2 * n + 1) ** 2

if sys.argv[2].lower() in ('yes', 'true', 't', 'y', '1'):
    prune = True
if sys.argv[2].lower() in ('no', 'false', 'f', 'n', '0'):
    prune = False

# Possible, relatively successful models
model1 = RandomForestClassifier(n_estimators=10)
model2 = GradientBoostingClassifier()
model3 = AdaBoostClassifier()
model4 = ExtraTreesClassifier()
model5 = BaggingClassifier()

def normalize_t2(data):
    """
    Normalizes a set of t2 images over the entire space of patient images. The rationale behind this normalization is
    to reduce images which are too bright and enhance images which are too dark onto one uniform scale for processing.
    
    :param data: The entire imaging dataset, loaded with matlab/scipy link
    :return: The normalized set of t2 images as a list of 256x256 numpy arrays
    """

    flat_data = []

    for i, patient in enumerate(data):
        if i in good_patients:
            t2 = patient[1]
            flat_data.append(t2.flatten())

    normalized_flat_data = normalize(flat_data)
    normalized_data = [np.reshape(datum, (256, 256)) for datum in normalized_flat_data]
    return normalized_data

def get_t2(data):
    """
    Fetch a list of t2 MRI data from the overall dataset. 
    
    :param data: The entire imaging dataset, loaded with matlab/scipy link
    :return: The set of all patients t2 images as a list of 256x256 numpy arrays.
    """
    return [patient[1] for i, patient in enumerate(data) if i in good_patients]

def get_adc(data):
    """
    Get tbe Apparent Diffusion Coefficient images from the overall dataset. 
    
    :param data: The entire imaging dataset, loaded with matlab/scipy link.
    :return: The set of all patients adc images as a list of 256x256 numpy arrays
    """
    return [patient[2] for i, patient in enumerate(data) if i in good_patients]

def get_dwi(data, weight="400"):
    """
    Get the diffusion weighted images from the overall dataset
    
    :param data: The entire imaging dataset, loadeed with matlab/scipy link
    :param weight: The weight of the dwi image; each patient has 5 possible weights.
    :return: The set of all patients dwi image at a specific weight as a list of 256x256 numpy arrays
    """
    wt_dict = {'10':3, '100':4, '400':5, '800':6, '2000':7}
    di = wt_dict[weight]
    return [patient[di] for i, patient in enumerate(data) if i in good_patients]

def get_filtered_t2(data, frequency, sigma_x=None, sigma_y=None):
    """
    Get Gabor filtered versions of the t2 MRI data.
    
    
    :param data: The entire imaging dataset, loadeed with matlab/scipy link
    :param frequency: The frequency of Gabor filter
    :param sigma_x: The x scale of Gabor filter
    :param sigma_y: The y scale of Gabor filter 
    :return: The set of all patients' gabor filtered t2 images as a list of 256x256 numpy arrays
    """

    filtered_data = []
    norm_data = normalize_t2(data)
    for patient_im in norm_data:
            filt_real, filt_t2 = gabor(patient_im, frequency=frequency, sigma_x=sigma_x, sigma_y=sigma_y)
            filtered_data.append(filt_real)
    return filtered_data

def get_masks(data):
    """
    Get mask data for the patients.
    
    :param data: The entire imaging dataset, loadeed with matlab/scipy link
    :return: The set of all patients' masks as a list of 256x256 numpy arrays
    """
    return [patient[0] for i, patient in enumerate(data) if i in good_patients]

def create_dataset(image, mask, n, prune=False, save_pxs=False):
    """
    Create a dataset for input into machine learning (X,y) from an image and a corresponding mask.
    The input data is gathered from the image, the output data (classification of pixel as cancerous or not) is gathered
    from the mask. 
    
    :param image: The data image to be transformed into X. 
    :param mask: The mask image to be transformed into y. 
    :param n: The number of pixels on each side of the pixel of interest (PIC) which will be included in the feature 
        vector. So if n=3, The total number of features is (3+1+3) * (3+1+3) = 49. 
    :param prune: If True, removes pixels which have feature vectors containing pixels outside the mask prostate area.
        If False, keeps all pixels in the prostate area, even if they are passing training data which is outside the 
        prostate. 
    :param save_pxs: Save pixel locations for image reconstruction later. 
    :return: X (a list of feature vectors of length (2*n + 1)^2), y (a list of mask classifications). 
    """
    row = 1
    subimages = []
    y = []
    acpx = []
    for i in range(n, 255-n):
        for j in range(n, 255-n):
            if mask[i,j]!=0:
                # numpy submatrices have (first index starting at 0):(last index starting from 1)

                store = True

                # dont use pixels with associated information outside the prostate
                if prune:
                    for v in mask[i-n:i+n+1, j-n:j+n+1]:
                        if 0 in v:
                            store = False
                            break

                # store the images as numpy arrays
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

    # flatten the matrices to feature vectors
    X = [image.flatten() for image in subimages]
    return X, y

def create_multiparametric_dataset(image_set, mask, n, prune=False, save_pxs=False):
    """
    Create a single patient dataset based on multiple images. For example, a set may include t2 data and adc data. This
    function extracts the necessary feature vectors from each image for a given patient and appends the feature vectors
    together, essentially creating a new, larger feature vector from multiple images. This is a good way of
    incorporating multiple imaging sources of data into the ML algorithm.
    
    This is similar to create_dataset except it uses multiple souces of image information. 
    
    :param image_set: A list of images for each patient as 256x256 numpy arrays. 
    :param mask: The mask for a single patient. 
    :param n: The number of pixels on each side of the pixel of interest (PIC) which will be included in the feature 
        vector. So if n=3, The total number of features is (3+1+3) * (3+1+3) = 49. 
    :param prune: If True, removes pixels which have feature vectors containing pixels outside the mask prostate area.
        If False, keeps all pixels in the prostate area, even if they are passing training data which is outside the 
        prostate. 
    :param save_pxs: Save pixel locations for image reconstruction later. 
    :return: X_new (input data as list of feature vectors), y (mask data classification as output info)
    """


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
    """
    Reconstruct a predicted image from a list of predictions and the original pixel locations.
    
    Note that if pruning, y_pred only classifies a subset of the entire mask, since some pixels around the outside
    are excluded from training and testing. 
    
    :param mask: The original mask, used to outline the prostate area for predictions.
    :param y_pred: The predicted values (classification of cancer) by the machine learning model. 
    :param pixels: The pixel locations (list of tuples) for the classifications in y_pred. 
    :return: 
    """

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

def runmodel(model, patient_index=-1, frequency=0.4, quiet=False, silent=False, normalized=False, filter_on=False,
             print_example_vector=False, save_reconstruction=False, adc_on=False, dwi_lvl=None):
    """
    Run a machine learning classifier. Trains on all data except one, and tests on one patient. 
    
    :param model: The sklearn-compatible model object to use for classification. 
    :param patient_index: The patient number (of available patients, 1-12) to be used for testing. 
    :param frequency: The gabor frequency to use if filter_on
    :param quiet: If True, suppresses intermediate info messages
    :param silent: If True, suppresses all info messages
    :param normalized: If True, uses normalized t2 instead of t2 as main dataset.
    :param filter_on: If True, add a Gabor-filtered dataset of the t2 to the data. 
    :param print_example_vector: If True, prints example vector (ie one point) being fed into the ML algorithm
    :param save_reconstruction: If True, saves the reconstructed images to .png files. 
    :param adc_on: If True, uses the apparent diffusion constant images as extra features. 
    :param dwi_lvl: If specified, uses the given weight to add diffusion weighted images to the data as extra features. 
    :return: 
    """

    modelname = model.__class__.__name__

    # Gather the types of data to be used in the predictions

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

    image_set_total = [t2]

    if filter_on:
        t2_filtered = get_filtered_t2(data, frequency=frequency)
        image_set_total.append(t2_filtered)

    if adc_on:
        adc = get_adc(data)
        image_set_total.append(adc)

    if dwi_lvl is not None:
        dwi = get_dwi(data, weight=dwi_lvl)
        image_set_total.append(dwi)

    X_train = []
    y_train = []

    # Create training data

    for i in range(n_patients):

        if i == patient_index:
            continue

        else:

            if len(image_set_total) > 1:
                image_set = [imset[i] for imset in image_set_total]
                X_train_single, y_train_single = create_multiparametric_dataset(image_set, masks[i], n, prune=prune)
            else:
                X_train_single, y_train_single = create_dataset(t2[i], masks[i], n, prune=prune)

            X_train += X_train_single
            y_train += y_train_single

    if print_example_vector:
        ex_vec = X_train[0]
        print "length of example vector:", len(ex_vec)
        print ex_vec

    # Create testing data

    if not quiet:
        print "{model}: patient {patient}/{n_patients}: making testing data".format(model=modelname, n_patients=n_patients, patient=patient_index+1)

    test_mask = masks[patient_index]
    test_t2 = t2[patient_index]

    if len(image_set_total) > 1:
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
    """
    Runs a cross validation loop over all available patients. 
    
    :param args: Positional arguments (to be passed to runmodel)
    :param kwargs: Keyword arguments (to be passed to runmodel)
    :return: 
    """

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
    # Lets run an example using T2 MRI, ADC, T2 MRI fitlered, and saving reconstructions!
    crossvalidate(model2, filter_on=True, adc_on=True, frequency=0.1, save_reconstruction=True)



    





