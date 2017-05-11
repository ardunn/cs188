import scipy.io
from matplotlib import pyplot as plt
import os

data = scipy.io.loadmat('data.mat')['data'][0]
os.getcwd()

labels = ['mask', 't2', 'adc', 'dwi0', 'dwi100', 'dwi400', 'dwi800', 'dwi2000']

for i, patient in enumerate(data):

    # create and navigate to the patient's folder
    patient_name = 'patient{}'.format(i)
    if not os.path.exists(patient_name):
        os.makedirs(patient_name)
    os.chdir(patient_name)

    print("making patient", i)

    # save the image
    for j, image in enumerate(patient):

        print ("saving image", j)
        plt.figure(i)
        image_name = 'patient{}_{}'.format(i, labels[j])
        plt.imshow(image)
        plt.savefig('{}.png'.format(image_name))
        plt.close()

    os.chdir('..')




