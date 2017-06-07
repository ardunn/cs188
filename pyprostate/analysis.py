"""
In this file we present functions for running high-level machine learning analyses for feature selecction, model
selection, and statistical importance. The functions in this file use the functions in ensemble.py (the most successful
algorithms we explored) as worker functions to actually perform machine learning analysis on the images. 

In general, the code below is used for generating and saving final results. 
"""

from ensemble import crossvalidate, model1, model2, model3, model4, model5
import pickle
import numpy as np


def model_influence():
    """
    Create and save results on model selection. 
    """
    results = {}

    for model in [model1, model2, model3, model4, model5]:
        auc = crossvalidate(model, quiet=True)
        results[str(model.__class__.__name__)] = auc

    pickle.dump(results, open('model_influence.p', 'wb'))

def frequency_influence():
    """
    Create and save results on the influence of freuqnecy on datasets with extra Gabor filters. 
    """
    results = {}
    for f in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        print "running with frequency of", f
        auc = crossvalidate(model2, filter_on=True, frequency=f, silent=True)
        results[f] = auc

    pickle.dump(results, open('frequency_influence.p', 'wb'))


def dwi_influence():
    """
    Create and save results on the influence of using different dwi weights as extra features. 
    :return: 
    """
    results = {}

    for dwi_lvl in ['10', '100', '400', '800', '2000']:
        print "dwi_level", dwi_lvl
        auc = crossvalidate(model2, filter_on=False, quiet=True, frequency=0.1, adc_on=False, dwi_lvl=dwi_lvl,
                            print_example_vector=False, silent=False, save_reconstruction=False)
        results[dwi_lvl] = auc

    pickle.dump(results, open('dwi_influence.p', 'wb'))


def multiparametric_influence():
    """
    Create and save resutls on the influence of different multiparametric data as extra features.
     
    The results of each run are block commented below the code which ran the run. 
    """


    # t2, filtered t2, adc, and best level of dwi
    res_all = crossvalidate(model2, filter_on=True, frequency=0.1, adc_on=True, dwi_lvl='2000')
    results = {'res_all': res_all}
    # result .718

    # t2, adc, best level of dwi
    res_nofilter = crossvalidate(model2, adc_on=True, dwi_lvl='2000')
    results['res_nofilter'] = res_nofilter
    # result .711

    # t2, filtered t2, best level of dwi
    res_noadc = crossvalidate(model2, filter_on=True, frequency=0.1, dwi_lvl='2000')
    results['res_noadc'] = res_noadc
    # result .6422

    # t2, filtered t2, adc
    res_nodwi = crossvalidate(model2, filter_on=True, frequency=0.1, adc_on=True)
    results['res_nodwi'] = res_nodwi
    # result .724

    pickle.dump(results, open('multiparametric_influence.p', 'wb'))


def statistical_final_run():
    """
    Run a final set of trials on our best model so far using the best models so far. 
    """

    results = []
    for i in range(10):
        results.append(crossvalidate(model2, adc_on=True, filter_on=True, frequency=0.1))

    results.append(crossvalidate(model2, adc_on=True, filter_on=True, frequency=0.1, save_reconstruction=True))
    mean_auc = np.mean(results)

    pickle.dump({'statistical_final_run': mean_auc}, open('statisical_final_run.p', 'wb'))


if __name__== "__main__":
    statistical_final_run()
