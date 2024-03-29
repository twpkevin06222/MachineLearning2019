import numpy as np
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import time

#---------------------------------------------------------------------------------------
#Argument Parser
parser = argparse.ArgumentParser(description='Input Parameters')
parser.add_argument('--data',  action = 'store', type = str, help = 'Please choose--random.csv or yacht.csv')
parser.add_argument('--learningRate', action = 'store', type = float,help = 'Please specify learning rate')
parser.add_argument('--threshold', action = 'store', type = float, help = 'Threshold as stopping criteria')
parser.add_argument('--dataVis', action = 'store', type = bool, help = 'Plot data correlations, by default FALSE', default= False)

#Input arguments
results = parser.parse_args()
lr = results.learningRate
input_path = results.data
thrs = results.threshold
dataVis = results.dataVis

#debugger
if lr==None and input_path==None and thrs==None:
    assert lr!=None and input_path !=None and thrs!=None, "Parser argument not specify! => (example)$ python linearregr.py --data /data/path/random.csv --learningRate 0.0001 --threshold 0.0001"
else:
    assert lr!=None, "Please specify learningRate in the command line!"
    assert input_path!=None, "Please specify input data set!"
    assert thrs!=None, "Please specify threshold in the command line!"

#---------------------------Helper functions------------------------------------------------
def set_array(inps):
    '''
    @param inps: data frame output from pandas library
    Converts dataframe into numpy array for easy work out
    '''
    inps = np.asarray(inps)
    # squeeze to remove indexing column from pandas
    return np.squeeze(inps)

def sq_loss(y_true, y_pred):
    '''
    Compute sum of squared loss
    @param y_true: array of target values/labels
    @param y_pred: array of predicted values

    return: sum of squared loss
    '''
    # squaring loss
    sq = (y_true - y_pred) ** 2
    return np.sum(sq)

def grad(x, y_true, y_pred):
    '''
    Compute the gradient of lost function
    @param x: array of data points
    @param y: data points of y_true
    @param y_pred: predicted target values

    return: gradient of shape [3,1]
    '''
    # x.shape [1000x3], error.shape [1000x1]
    # hence written as np.dot(x^T,error)
    return np.dot(x.transpose(), (y_true - y_pred))

def grid_plot_2col(x, y, x_col, marker='kx', figsize=(10, 18)):
    '''
    Plot grid of figures with 2 columns by default
    @param x: x array
    @param y: y array
    @param x_col: number of columns for x data set
    @param marker: Data points marker shape in plot
    @param figsize: Figure size of grid plot output

    return: A figure containing grid plot
    '''
    # if number of x columns is odd number
    if x_col % 2 != 0:
        plot_row = int((x_col + 1) / 2)
    # if number of x columns is even number
    else:
        plot_row = int(x_col / 2)

    fig, axes = plt.subplots(plot_row, 2, figsize=figsize)
    fig.suptitle("Correlations between Data Points and Target Values", fontsize=16)

    if x_col % 2 != 0:
        # delete last axes since x columns is odd number
        fig.delaxes(axes[plot_row - 1, 1])
        for i, ax in enumerate(axes.flat):
            if i == x_col: break
            ax.plot(x[:, i], y, marker)
            ax.set_xlabel('x_{}'.format(i))
            ax.set_ylabel('y')
    else:
        for i, ax in enumerate(axes.flat):
            ax.plot(x[:, i], y, marker)
            ax.set_xlabel('x_{}'.format(i))
            ax.set_ylabel('y')

    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()

#----------------------------Data preprocessing---------------------------------------
input_ds = pd.read_csv(input_path, header = None)
#insert initial bias vector in the first column
input_ds.insert(0,None,np.ones(input_ds.shape[0]))
n_cols = len(input_ds.columns)

#Vectorization of input data
#from the first column to second last column is assign as x values
x = input_ds.iloc[:,0:n_cols-1]
#last column is the target values which is y
y = input_ds.iloc[:,n_cols-1]

#----------------------------Main------------------------------------------------------
#convert data frames to numpy arrays
x = set_array(x)
y = set_array(y)

#Gradient Descent
def main():
    iters = 0
    loss_list = [] #list to store SSE at each iteration for comparison
    error = 1000 #random value for initializing error sufficiently larger than stopping criteria

    print()
    # dynamic print heading
    print("{:<6}".format("Iter"), end='')
    for i in range(n_cols - 1):
        print("{:^8}".format('w_' + str(i)), end='')
    print("{:>9}".format('SSE'))

    # measure execution time
    ex_start = time.process_time()
    # stopping criteria
    while error >= thrs:
        if iters == 0:
            # initialize with all weights as 0
            w = np.zeros(n_cols-1)
        else:
            # update each weights respectively
            w += lr * grad(x, y, y_pred)

        # y_pred = w.x with => x.shape [1000x3], w.shape [3x1]
        # hence written as x.w
        y_pred = np.dot(x, w)
        loss = sq_loss(y, y_pred)
        loss_list.append(loss)

        if iters == 0:
            error = loss
        else:
            # stopping criteria if loss from previous step minus
            # loss from current step is below the threshold
            error = loss_list[iters-1] - loss_list[iters]

        # dynamic print values
        print("{:^6}".format(iters), end='')
        for i in range(n_cols-1):
            print("{:^8.4f}".format(w[i].round(4)), end='')
        print("{:>11.4f}".format(loss.round(4)))

        # update iteration
        iters += 1

    #end execution time and calculate time elapsed
    ex_elapsed = (time.process_time() - ex_start)
    print()
    print("--End of iteration! After: {} iterations--".format(iters))
    print("--Execution time: {} s --".format(ex_elapsed))

    # Data Visualization (optional)
    if dataVis == True:
        grid_plot_2col(x, y, n_cols - 1)
#-----------------------------------Run--------------------------------------
if __name__=="__main__":
    main()
