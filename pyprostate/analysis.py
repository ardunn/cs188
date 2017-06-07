from ensemble import crossvalidate, model1, model2, model3, model4, model5, model6
import pickle
import numpy as np


def model_influence():
    results = {}

    for model in [model1, model2, model3, model4, model5, model6]:
        auc = crossvalidate(model, quiet=True)
        results[str(model.__class__.__name__)] = auc

    pickle.dump(results, open('model_influence.p', 'wb'))

def frequency_influence():
    results = {}
    for f in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        print "running with frequency of", f
        auc = crossvalidate(model2, filter_on=True, frequency=f, silent=True)
        results[f] = auc

    pickle.dump(results, open('frequency_influence.p', 'wb'))


def dwi_influence():
    results = {}

    for dwi_lvl in ['10', '100', '400', '800', '2000']:
        print "dwi_level", dwi_lvl
        auc = crossvalidate(model2, filter_on=False, quiet=True, frequency=0.1, adc_on=False, dwi_lvl=dwi_lvl,
                            print_example_vector=False, silent=False, save_reconstruction=False)
        results[dwi_lvl] = auc

    pickle.dump(results, open('dwi_influence.p', 'wb'))


def multiparametric_influence():
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

    results = []
    for i in range(10):
        results.append(crossvalidate(model2, adc_on=True, filter_on=True, frequency=0.1))

    results.append(crossvalidate(model2, adc_on=True, filter_on=True, frequency=0.1, save_reconstruction=True))
    mean_auc = np.mean(results)

    pickle.dump({'statistical_final_run': mean_auc}, open('statisical_final_run.p', 'wb'))


if __name__== "__main__":
    frequency_influence()
