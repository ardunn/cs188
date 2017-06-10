## Welcome to Alex and Breanna's CS 188 Project

Our project and website for CS 188: Medical Imaging under Fabient Scalzo. Our project focused on applying machine learning to prostate cancer diagnosis. 

### Structure
```shell
cs188/              # -> Root folder
├── docs/           # -> Content for our blogposts on our website
└── pyprostate/     # -> Project code
    └── graphics/   # -> MRI images
    └── analysis.py
    └── ensemble.py
    └── first_attempt.py
    └── neural_network.py
```

### Setup 

Install virtualenv: 

` $ [sudo] pip install virtualenv `

Activate the virtual environment to install the necessary dependencies: 

` $ source env_188/bin/activate `

### Usage

In `ensemble.py`, the main function runs our `crossvalidate` function which executes our machine learning model with specific settings. 

`ensemble.py` takes in the following arguments: 

```bash
$ python ensemble.py <n> <yes/true/t/y/1 or no/false/f/n/0> 
```

`n` specifies the size of the subimages.

Currently, `ensemble.py` will run the following: 

```python
# Lets run an example using T2 MRI, ADC, T2 MRI fitlered, and saving reconstructions!
crossvalidate(model2, filter_on=True, adc_on=True, frequency=0.1, save_reconstruction=True)
```

On the command line run:

```bash
$ python ensemble.py 6 y 
```

to see our results!

### Settings
Here are the following keyword arguments you can pass into `crossvalidate`: 

* frequency: The gabor frequency to use if filter_on
* quiet: If True, suppresses intermediate info messages
* silent: If True, suppresses all info messages
* normalized: If True, uses normalized t2 instead of t2 as main dataset.
* filter_on: If True, add a Gabor-filtered dataset of the t2 to the data. 
* print_example_vector: If True, prints example vector (ie one point) being fed into the ML algorithm
* save_reconstruction: If True, saves the reconstructed images to .png files. 
* adc_on: If True, uses the apparent diffusion constant images as extra features. 
* dwi_lvl: If specified, uses the given weight to add diffusion weighted images to the data as extra features. 
   
