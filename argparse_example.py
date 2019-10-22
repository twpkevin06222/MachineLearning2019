import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Short sample')

parser.add_argument('--data',  action = 'store',dest ='ds',help = 'Input data sets')
parser.add_argument('--learningRate', action = 'store', dest = 'lr', help = 'Please specify learning rate')
parser.add_argument('--threshold', action = 'store', dest = 'thrs', help = 'Threshold as stopping criteria')


results = parser.parse_args()

lr = results.lr
input_ds = results.ds
thrs = results.thrs

print("Learning rate is", lr)
print("The input data set is", input_ds)
print("The threshold is", thrs)

