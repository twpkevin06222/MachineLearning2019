import numpy as np
import pandas as pd
import time
from math import log
import xml.etree.ElementTree as ET
import argparse

#--------------------------- Argument Parser----------------------------------
parser = argparse.ArgumentParser(description='Input Parameters')
parser.add_argument('--data',  action = 'store', type = str, help = 'Please choose--car.csv or nursery.csv')
parser.add_argument('--output',  action = 'store', type = str, help = 'Please name file and specify where .xml file is saved')

#Input arguments
results = parser.parse_args()
input_path = results.data
output_xml = results.output

#----------------------------Helper Functions----------------------------------
def entropy(target_col, n_class):
    """
    This function calculates the entropy of the dataset

    @param target_col: The column where the target values are stored
    @param n_class: For log base

    return: entropy of the target values w.r.t the dataset
    """
    values, counts = np.unique(target_col, return_counts=True)
    entropy = np.sum([(-counts[i] / np.sum(counts)) * log((counts[i] / np.sum(counts)), n_class) for i in range(len(values))])
    return entropy


def info_gain(data, split_attribute_name, n_class, target_name):
    """
    This function computes the information gain of a feature by subtracting total entropy with weighted
    entropy of the values in the feature respectively

    @param data: input data set
    @param split_attribute_name: feature column
    @param n_class: for log base
    @target_name: name of target column

    return: information gain
    """
    # Compute the entropy of the original dataset
    total_entropy = entropy(data[target_name], n_class)

    # Calculate the values and the corresponding counts for the split attribute
    vals, counts = np.unique(data[split_attribute_name], return_counts=True)

    # Calculate the weighted entropy
    Weighted_Entropy = np.sum([(counts[i] / np.sum(counts)) * entropy(data.where(data[split_attribute_name] == vals[i]).dropna()[target_name], n_class) for i in range(len(vals))])

    # Calculate the information gain by subtracting weighted entropy from total entropy
    return total_entropy - Weighted_Entropy


def ID3_xml(data, originaldata, features, n_class, target_attribute_name, tree_xml, best_feature=None,
            value=None, parent_node_class=None, space=''):
    """
    This function compute the ID3 algorithm of a decision tree

    @param data: Data that the algorithm is currently running
    @param original_data: Original dataset that includes all the feature columns
    @param features: A list containing feature column names
    @param n_class: Number of class as log base
    @param target_attribute_name: Column name where the target values are stored.
    @param best_feature: Best feature used at the particular iteration
    @param value: The value of the best feature used at the particular iteration
    @param parent_node_class: The best target feature value will be stored

    return: Tree structure in xml format
    reference: https://www.python-course.eu/Decision_Trees.php
    """
    # Stopping criteria for creating a leaf node
    # If all target_values have the same value, return this value, because entropy will be 0
    if len(np.unique(data[target_attribute_name])) <= 1:
        space += ' '
        ent = entropy(data[target_attribute_name], n_class)
        target_val = np.unique(data[target_attribute_name])[0]
        # leaf node
        # here tree_xml<=sub_tree from recursive function
        sub_sub_tree = ET.SubElement(tree_xml, 'node', entropy=str(ent), feature=str(best_feature), value=str(value))
        sub_sub_tree.text = str(target_val)
        return target_val

    # Return the mode target feature value in the original dataset if the dataset is empty
    elif len(data) == 0:
        # axis 1 is the list where the counts are stored
        return np.unique(originaldata[target_attribute_name])[np.argmax(np.unique(originaldata[target_attribute_name], return_counts=True)[1])]

    # If the feature space is empty, return the mode target feature value of the direct parent node
    elif len(features) == 0:
        return parent_node_class

    # Grow tree
    else:
        if (value == None):
            sub_tree = tree_xml
        else:
            ent = entropy(data[target_attribute_name], n_class)
            # root node
            sub_tree = ET.SubElement(tree_xml, 'node', entropy=str(ent), feature=str(best_feature), value=str(value))
        # Set the default value for parent node
        parent_node_class = np.unique(data[target_attribute_name])[np.argmax(np.unique(data[target_attribute_name], return_counts=True)[1])]
        # Compute the gain of each feature respectively
        item_values = [info_gain(data, feature, n_class, target_attribute_name) for feature in features]  # Return the information gain values for the features in the dataset
        # retrieving the index of the highest gain feature for best feature
        best_feature_index = np.argmax(item_values)
        best_feature = features[best_feature_index]

        # Remove(isolate) the feature with the best information gain from the feature space
        # because we are sorting values w.r.t the best feature
        features = [i for i in features if i != best_feature]
        # Grow a branch under the root node for each possible value of the root node feature
        for value in np.unique(data[best_feature]):
            # Split the dataset along the value of the feature with the largest information gain and create sub_datasets
            sub_data = data.where(data[best_feature] == value).dropna()

            # Recursively compute the ID3 algorithm for each of those sub datasets with the new parameters
            subtree = ID3_xml(sub_data, data, features, n_class, target_attribute_name, sub_tree,
                              best_feature, value, parent_node_class, space)

        return tree_xml

# ----------------------------Data preprocessing------------------------------------------
input_ds = pd.read_csv(input_path)
#number of columns
n_cols = len(input_ds.columns)
#naming the columns
#initiate empty list for column name
col_name = []
for i in range(n_cols):
    #last column is the target value
    if (i == n_cols-1):
        col_name.append('class')
    else:
        col_name.append('att{}'.format(i))
#assigning column names to data set
input_ds.columns = col_name
#number of class
n_class = len(set(input_ds['class']))

#----------------------------Main------------------------------------------------------
#id3 algorithm for decision tree
def main():
    # measure execution time
    ex_start = time.process_time()
    #comopute initial entropy for the storing in the root node
    init_ent = entropy(input_ds['class'], n_class)
    tree_xml = ET.Element('tree', entropy=str(init_ent))
    tree_out = ID3_xml(input_ds, input_ds, input_ds.columns[:-1], n_class, "class", tree_xml)
    #end execution time and calculate time elapsed
    ex_elapsed = (time.process_time() - ex_start)
    # write in xml style
    xmlWrite = ET.ElementTree(tree_out)
    xmlWrite.write(output_xml)
    print('-- End of execution! Your .xml file is saved as: {} --'.format(output_xml))
    print("--Execution time: {} s --".format(ex_elapsed))

#-----------------------------------Run------------------------------------------------
if __name__=="__main__":
    main()