import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
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
def y_pred(w, x):
    '''
    This function calculate the predicted labels

    @param w: weight vectors
    @param x: feature arrays

    return: array of predicted labels where y = 1 if w.x>0, else 0
    '''
    w_x = np.dot(x, w)
    return np.where(w_x > 0, 1, 0)

def error_rate(y, y_p):
    '''
    This function calculate the total number of
    misclassified labels

    @param y: true labels
    @param y_p: predicted labels

    return: The sum of misclassified labels at the
            particular iteration
    '''
    misclassified = np.abs(y - y_p)
    return np.sum(misclassified)

def anneal_lr(lr, iters):
    '''
    This function calculates the annealing learning rate

    @param lr: initial learning rate
    @param iters: iteration

    return: damped learning rate at the particular
            iteration
    '''
    # initial learning rate at initial iteration
    if (iters == 0):
        return lr
    else:
        lr /= iters
        return lr

def lr_switch(lr, iters, toggle):
    '''
    This function acts as a switch to alternate between
    two different learning rate

    @param lr: initial learning rate
    @param iters: iteration
    @param toggle: constant learning rate if 0
                   else anneal learning rate
    '''
    if (toggle == 0):
        return lr
    else:
        return anneal_lr(lr, iters)

# ----------------------------Data preprocessing------------------------------------------
#read data using pandas library as .csv
input_ds = pd.read_csv(input_path, header = None, sep="\t")
n_cols_0 = len(input_ds.columns)
if n_cols_0>3:
    input_ds = input_ds.drop(n_cols_0-1, axis=1) #delete the last column because its NaN for all rows
#insert bias
input_ds.insert(1,None,np.ones(input_ds.shape[0]))
#actual number of columns
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
#setting up label vector, if A then 1 else 0
y = np.where(input_ds['labels']=='A',1,0)
#setting up features array
x = input_ds.iloc[:,1:n_cols]
x = np.squeeze(np.asarray(x))

#----------------------------Main------------------------------------------------------
#single perceptron algorithm
lr_0 = 1 #initial learning rate
lr_variant = 2 #learning rate variant
error_variant_list = [] #error list to store error for different variant
y_pred_list = [] #predicted label list to store predicted label for different variant
w_variant_list = [] # weight list to store weights for different variant
def main():
    for variant in range(lr_variant):
        iters = 0 #initial iteration
        iters_max = 100 #max iteration
        error_list = []

        while iters <= iters_max:
            if (iters == 0):
                w = np.zeros(n_cols - 1)
            else:
                w += lr_switch(lr_0, iters, variant) * np.dot(x.transpose(), (y - y_p))

            y_p = y_pred(w, x)
            error = error_rate(y, y_p)
            error_list.append(error)
            #update iterations
            iters += 1
        error_variant_list.append(error_list)
        y_pred_list.append(y_p)
        w_variant_list.append(w)

    with open(output_tsv, 'wt') as write_tsv:
        tsv_writer = csv.writer(write_tsv, delimiter='\t')
        for i in range(len(error_variant_list)):
            tsv_writer.writerow(error_variant_list[i])

    print('-- End of execution! Your .tsv file is saved as: {} --'.format(output_tsv))
#-----------------------------------Run------------------------------------------------
if __name__=="__main__":
    main()

