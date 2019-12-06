import numpy as np
import csv
import pandas as pd
from math import pi, exp, sqrt
import argparse

#--------------------------- Argument Parser----------------------------------
parser = argparse.ArgumentParser(description='Input Parameters')
parser.add_argument('--data',  action = 'store', type = str, help = 'Please choose--Example.tsv or Gauss.tsv')
parser.add_argument('--output',  action = 'store', type = str, help = 'Please name file and specify where .tsv file is saved')

#Input arguments
results = parser.parse_args()
input_path = results.data
output_tsv= results.output

#----------------------------Helper Functions----------------------------------
def retrieve_class_index(class_col, vals):
    '''
    Retrieve the class index w.r.t the class row

    @param class_col: The column or array where the class is stored
    @param vals: list containing the classes

    return a list of index
    '''
    index_list = []
    for val in vals:
        index_array = np.where(class_col == val)
        index_list.append(index_array)
    return index_list


def mean(x, index_list, counts):
    '''
    Compute the mean w.r.t class and attribute

    @param x: The array of where x attributes is stored
    @param index_list: list of index w.r.t the class
    @param counts: the number of particular class

    return: An array of mean with size (n_class, n_attribute)
    '''
    class_means = []
    for i in range(len(index_list)):
        # sum the x attributes in rows w.r.t the same class index
        x_sum = np.sum([x[idx] for idx in index_list[i]], axis=1)
        class_mean = x_sum / counts[i]
        class_means.append(class_mean)
    return np.squeeze(np.asarray(class_means))


def variance(x, index_list, mean_arr, counts):
    '''
    Compute the variance (sigma**2) w.r.t class and attribute

    @param x: The array of where x attributes is stored
    @param index_list: list of index w.r.t the class
    @param mean_arr: Array of mean with size (n_class, n_attribute)
    @param counts: the number of particular class

    return: An array of variance with size (n_class, n_attribute)
    '''
    variance_list = []
    for i in range(len(index_list)):
        # retrieve x attributes w.r.t the same class index
        x_class = np.array([x[idx] for idx in index_list[i]])
        sum_func = np.sum((x_class - mean_arr[i]) ** 2, axis=1)
        variance_class = sum_func / (counts[i] - 1.0)
        variance_list.append(variance_class)
    return np.squeeze(np.asarray(variance_list))


def likelihood(x, mean, variance):
    '''
    Compute the likelihood, which is the joint probability
    p(a|c), the probability of attribute x, given class c

    @param x: x attribute
    @param mean: mean value
    @param variance: variance value

    return: likelihood
    '''
    denominator = sqrt(2 * pi * variance)
    exp_term = exp(-(((x - mean) ** 2) / (2 * variance)))
    return exp_term / denominator


def prob_class(class_col, counts):
    '''
    Calculate the probability of each class respectively

    @param class_col: The column of class data being stored

    return: list of class probability
    '''
    class_probs = []  # list to store class means
    for count in counts:
        class_prob = count / len(class_col)
        class_probs.append(class_prob)
    return class_probs

# ----------------------------Data preprocessing------------------------------------------
#read data using pandas library as .csv
input_ds = pd.read_csv(input_path, header = None, sep="\t")
#drop columns if the columns consisted of NaN
input_ds = input_ds.dropna(axis='columns')
n_cols = len(input_ds.columns)
# initiate empty list for column name
col_name = []
# naming scheme starts with labels because first column is the label column
# feature columns will be name as x_{}.format(n_cols)
for i in range(n_cols):
    # last column is the target value
    if (i == 0):
        col_name.append('labels')
    else:
        col_name.append('x_{}'.format(i - 1))

#assigning column names to data set
input_ds.columns = col_name
#setting up class vector
c = np.squeeze(np.asarray(input_ds['labels']))
#setting up features array
x = input_ds.iloc[:,1:n_cols]
x = np.squeeze(np.asarray(x))

#----------------------------Main------------------------------------------------------
#gaussian naive bayes algorithm
def main():
    vals, counts = np.unique(input_ds['labels'], return_counts=True)
    index_list = retrieve_class_index(c, vals)
    prior_arr = prob_class(c, counts)
    mean_arr = mean(x, index_list, counts)
    variance_arr = variance(x, index_list, mean_arr, counts)
    n_clc = mean_arr.shape[0]
    n_att = mean_arr.shape[1]
    argmax_list = []
    #start multiple loop
    idx_argmax_list = []
    for idx in range(len(x)):
        posterior_list = []
        for clc in range(n_clc):
            total_likelihood = 1
            for att in range(n_att):
                total_likelihood *= likelihood(x[idx, att], mean_arr[clc, att], variance_arr[clc, att])
            posterior_list.append(prior_arr[clc] * total_likelihood)
        idx_argmax = np.argmax(posterior_list)
        idx_argmax_list.append(idx_argmax)
    argmax_list.append(idx_argmax_list)
    argmax_list = np.squeeze(np.asarray(argmax_list))  # convert tuple to array
    argmax_list = np.reshape((argmax_list), (-1))  # flatten the array
    # map the idx to the class for comparison
    map_idx2class = np.array([vals[idx] for idx in argmax_list])
    comparator_arr = np.array([map_idx2class[idx] == c[idx] for idx in range(len(c))])
    #compute missclassified
    missclassified = np.sum(np.where(comparator_arr == False, 1, 0))

    #write .tsv
    output_total = []
    for clc in range(n_clc):
        output_list_class = []
        for att in range(n_att):
            output_list_class.append(mean_arr[clc, att])
            output_list_class.append(variance_arr[clc, att])
        output_list_class.append(prior_arr[clc])
        output_total.append(output_list_class)
    output_total = np.squeeze(np.asarray(output_total))

    with open(output_tsv, 'wt') as write_tsv:
        tsv_writer = csv.writer(write_tsv, delimiter='\t')
        for i in range(len(output_total)):
            tsv_writer.writerow(output_total[i])
        tsv_writer.writerow([missclassified])
    write_tsv.close()

    print('-- End of execution! Your .tsv file is saved as: {} --'.format(output_tsv))
#-----------------------------------Run------------------------------------------------
if __name__=="__main__":
    main()

