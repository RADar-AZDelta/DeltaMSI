#!/bin/python3.9
# DeltaMSI: AI-based screening for microsatellite instability in solid tumors
# Copyright (C) 2022  Koen Swaerts, AZ Delta

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import seaborn as sns
from statistics import mean, stdev
from random import shuffle

from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import ConfusionMatrixDisplay
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore')

# train-test split
def train_test_split(sample_list: list, test_perc: float=0.33, verbose: bool=False):
    """Split the dataset in a train and test set, based on the samples and the ihc information

    Args:
        sample_list (list): A list of all samples
        test_perc (float, optional): The percentage of the size for the test set. Defaults to 0.33.
        verbose (bool, optional): Print user info. Defaults to False.

    Returns:
        (list, list, list, list): positive train set, negative train set, positive test set, negative test set
    """
    if verbose:
        print("Start the train-test split with {} samples".format(len(sample_list)))
    pos_list = list()
    neg_list = list()
    for sample in sample_list:
        if sample.ihc:
            pos_list.append(sample)
        else:
            neg_list.append(sample)
    if verbose:
        print("Found {} positives, {} negatives".format(len(pos_list), len(neg_list)))
    pos_split = int(len(pos_list) * test_perc) + 1
    neg_split = int(len(neg_list) * test_perc) + 1
    shuffle(pos_list)
    shuffle(neg_list)
    pos_train = list()
    pos_train.extend(pos_list[pos_split:])
    pos_test = list()
    pos_test.extend(pos_list[:pos_split])
    neg_train = list()
    neg_train.extend(neg_list[neg_split:])
    neg_test = list()
    neg_test.extend(neg_list[:neg_split])
    if verbose:
        print("Split {} positive samples in {} train and {} test".format(len(pos_list),
                                                len(pos_train), len(pos_test)))
        print("Split {} negative samples in {} train and {} test".format(len(neg_list),
                                                len(neg_train), len(neg_test)))
    return (pos_train, neg_train, pos_test, neg_test)

# creation of k-folds
def k_fold(pos_list: list, neg_list: list, split: int, verbose: bool=False) -> dict:
    """Create a dict with the kfolds: shuffled, stratified split of samples

    Args:
        pos_list (list): The positive list to use
        neg_list (list): The negative list to use
        split (int): The number of splits to make
        verbose (bool, optional): Print user output. Defaults to False.

    Returns:
        dict: The different folds with keys: the number of the fold, values: dict with 
        "pos_train, neg_train, pos_test and neg_test" sample lists 
    """
    # split the positive and negative list into split folds (split=k)
    # this will return a shuffled, stratified split
    # return value is a dict, with the number of the fold as key, values are:
    # pos_train, neg_train, pos_test, neg_test
    if verbose:
        print("Started with {} positive and {} negative samples".format(len(pos_list), len(neg_list)))
    pos_split = int(len(pos_list)/split)
    neg_split = int(len(neg_list)/split)
    shuffle(pos_list)
    shuffle(neg_list)
    pos_split_list = list()
    neg_split_list = list()
    for s in range(0, split):
        # split in different ranges
        if s == split-1:
            pos_split_list.append(pos_list[s*pos_split:])
            neg_split_list.append(neg_list[s*neg_split:])
        else:
            pos_split_list.append(pos_list[s*pos_split:(s+1)*pos_split])
            neg_split_list.append(neg_list[s*neg_split:(s+1)*neg_split])
    # create dict with the folds, and samples
    k_fold_dict = dict()
    for s in range(0, split):
        k_fold_dict[s] = dict()
        k_fold_dict[s]["pos_train"] = list()
        k_fold_dict[s]["neg_train"] = list()
        k_fold_dict[s]["pos_test"] = list()
        k_fold_dict[s]["neg_test"] = list()
        for i in range(0, split):
            if i == s:
                # found test set
                k_fold_dict[s]["pos_test"] = pos_split_list[i]
                k_fold_dict[s]["neg_test"] = neg_split_list[i]
            else:
                k_fold_dict[s]["pos_train"].extend(pos_split_list[i])
                k_fold_dict[s]["neg_train"].extend(neg_split_list[i])
    if verbose:
        print("Split {} folds in:".format(split))
        for s in range(0, split):
            print("Fold {}".format(s))
            print("\tTrain: {} positive, {} negative".format(len(k_fold_dict[s]["pos_train"]),
                                                            len(k_fold_dict[s]["neg_train"])))
            print("\tValidate: {} positive, {} negative".format(len(k_fold_dict[s]["pos_test"]),
                                                           len(k_fold_dict[s]["neg_test"])))
    return k_fold_dict

# creation of dataframes, based on regions
def create_dataframe_of_region(region_name: str, sample_list1: list, sample_list2: list=None, depth: int=0, add_ihc: bool=True):
    """Create a dataframe based on the given sample lists and region name

    Args:
        region_name (str): The name of the region to use
        sample_list1 (list): A list of samples
        sample_list2 (list, optional): A second list of samples (eg if pos and neg are 2 seperate lists). Defaults to None.
        depth (int, optional): The depth to use to select the . Defaults to 0.
        add_ihc (bool, optional): Add ihc information (for training only). Defaults to True.

    Returns:
        pandas.DataFrame: The dataframe with the requested values
    """
    my_dict = dict()
    for i in range(0, sample_list1[0].get_region(region_name).region_length+1):
        my_dict["L{}".format(i)] = list()
    if add_ihc:
        my_dict["ihc"] = list()
    for sample in sample_list1:
        region = sample.get_region(region_name)
        if region.depth >= depth:
            for i in range(0, region.region_length+1):
                my_dict["L{}".format(i)].append(region.get_value_norm_on(i, flex_cutoff=True))
            if add_ihc:
                if sample.ihc:
                    my_dict["ihc"].append(1)
                else:
                    my_dict["ihc"].append(0)
    if sample_list2 is not None:
        for sample in sample_list2:
            region = sample.get_region(region_name)
            if region.depth >= depth:
                for i in range(0, region.region_length+1):
                    my_dict["L{}".format(i)].append(region.get_value_norm_on(i, flex_cutoff=True))
                if add_ihc:
                    if sample.ihc:
                        my_dict["ihc"].append(1)
                    else:
                        my_dict["ihc"].append(0)
    df = pd.DataFrame.from_dict(my_dict)
    return df

def extend_parameters(parameter_dict: dict, param_list: list=None) -> list:
    """Creates a list with all parameter combinations in the given dict

    Args:
        parameter_dict (dict): The dict with for each parameter a list of possible values
        param_list (list, optional): The created list up till now (recursive method). Defaults to None.

    Returns:
        list: The list with all possible combinations
    """
    if len(parameter_dict) == 0:
        return param_list
    param_name = list(parameter_dict)[0]
    if param_list is None:
        param_list = list()
        for val in parameter_dict[param_name]:
            p_dict = dict()
            p_dict[param_name] = val
            param_list.append(p_dict)
    else:
        new_param_list = list()
        for p_dict in param_list:
            for val in parameter_dict[param_name]:
                new_dict = p_dict.copy()
                new_dict[param_name] = val
                new_param_list.append(new_dict)
        param_list = new_param_list
    del parameter_dict[param_name]
    return extend_parameters(parameter_dict, param_list)

def parameter_tuning(model_class, parameter_list: list, k_fold_dict: dict, region_name: str, 
                     use_positives: bool=True, depth: int=30, verbose: bool=False):
    """Do a grid search

    Args:
        model_class (class): The class of the model (not an object!)
        parameter_list (list): a list of all dicts with possible parameters
        k_fold_dict (dict): The prepared kfolds dict
        region_name (str): The name of the region to optimase
        use_positives (bool, optional): Use positive samples (for outlier detection methods). Defaults to True.
        depth (int, optional): The depth for a region to use. Defaults to 30.
        verbose (bool, optional): Print user output. Defaults to False.

    Returns:
        (int, list): the best score, the list of best scoring parameters
    """
    # extend the parameter list for grid search
    extended_parameter_list = list()
    for parameter_dict in parameter_list:
        extended_parameter_list.extend(extend_parameters(parameter_dict))
    if len(extended_parameter_list) == 0:
        extended_parameter_list.append(dict())
    # create the dataframes from the k_fold_dict and region
    my_k_fold_dict = dict()
    for fold in k_fold_dict:
        fold_dict = k_fold_dict[fold]
        my_k_fold_dict[fold] = dict()
        s_list = list()
        s_list.extend(fold_dict["neg_train"])
        if use_positives:
            s_list.extend(fold_dict["pos_train"])
        df = create_dataframe_of_region(region_name, 
                                     s_list, depth=depth)
        my_k_fold_dict[fold]["X_train"] = df.drop(["ihc"], axis=1)
        my_k_fold_dict[fold]["y_train"] = df["ihc"]
        s_list = list()
        s_list.extend(fold_dict["neg_test"])
        if use_positives:
            s_list.extend(fold_dict["pos_test"])
        df = create_dataframe_of_region(region_name, 
                                     s_list, depth=depth)
        my_k_fold_dict[fold]["X_test"] = df.drop(["ihc"], axis=1)
        my_k_fold_dict[fold]["y_test"] = df["ihc"]
    # do the hyperparameter tuning
    highest_acc = 0
    best_parameters = list()
    for parameters in tqdm(extended_parameter_list, disable=not verbose):
        bacc_list = list()
        for k in my_k_fold_dict:
            X_train = my_k_fold_dict[k]["X_train"]
            y_train = my_k_fold_dict[k]["y_train"]
            X_test = my_k_fold_dict[k]["X_test"]
            y_test = my_k_fold_dict[k]["y_test"]
            try:
                model = model_class(**parameters)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                if -1 in y_pred:
                    # inliers are labeled as 1, outliers as -1
                    # which needs to be converted to 0 as normal, 1 as abnormal
                    y_pred = np.where(y_pred==1, 0, y_pred)
                    y_pred = np.where(y_pred==-1, 1, y_pred)
                bacc_list.append(balanced_accuracy_score(y_test, y_pred))
            except:
                pass
            
        if len(bacc_list) > 1:
            new_acc = mean(bacc_list)
        else:
            new_acc = 0
        if len(best_parameters) == 0:
            highest_acc = new_acc
            best_parameters = parameters
        if new_acc > highest_acc:
            highest_acc = new_acc
            best_parameters = parameters
    if verbose:
        print("Highest accuracy is {}".format(highest_acc))
        print("Best parameters are: {}".format(best_parameters))
    return (highest_acc, best_parameters)

def create_model(model_class, parameters: list, X_train, y_train):
    """Creates and trains a model

    Args:
        model_class (class): The class of the model (not an object)
        parameters (list): The list of parameters to use
        X_train (pandas.DataFrame): The dataframe to use for training
        y_train (pandas.Series): The labels

    Returns:
        model object: The trained model
    """
    model = model_class(**parameters)
    model.fit(X_train, y_train)
    return model

def plot_confusion_matrix(y_test: list, y_pred: list):
    """Plot the confusion matrix

    Args:
        y_test (list): The actual target values
        y_pred (list): The predicted values
    """
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix(y_test, y_pred,
                                                               normalize='true'),
                              display_labels=["Neg", "Pos"])
    disp = disp.plot(include_values=["Neg", "Pos"])
    plt.show()
    
def evaluate(model_class, parameters: list, X_train, y_train, X_test, y_test, X_comp, y_comp, verbose: bool=False):
    """Evaluate the performance of the given model

    Args:
        model_class (class): The class of the model (not an object!)
        parameters (list): The list of hyperparameters
        X_train (pandas.DataFrame): The dataframe to train the model (for outliers without the positives)
        y_train (pandas.Series): The targets for the training
        X_test (pandas.DataFrame): The dataframe of the test samples
        y_test (pandas.Series): The targets for testing
        X_comp (pandas.DataFrame): The complete training dataframe (identical to X_train, but for outliers with the positives)
        y_comp (pandas.Series): The complete targets
        verbose (bool, optional): Print user output (if False, same as create_model). Defaults to False.

    Returns:
        model object: The trained model
    """
    if verbose:
        print("Evaluation")
        print("Start Training")
    model = create_model(model_class, parameters, X_train, y_train)
    y_pred_test = model.predict(X_test)
    if -1 in y_pred_test:
        # inliers are labeled as 1, outliers as -1
        # which needs to be converted to 0 as normal, 1 as abnormal
        y_pred_test = np.where(y_pred_test==1, 0, y_pred_test)
        y_pred_test = np.where(y_pred_test==-1, 1, y_pred_test)
    y_pred_train = model.predict(X_comp)
    if -1 in y_pred_train:
        # inliers are labeled as 1, outliers as -1
        # which needs to be converted to 0 as normal, 1 as abnormal
        y_pred_train = np.where(y_pred_train==1, 0, y_pred_train)
        y_pred_train = np.where(y_pred_train==-1, 1, y_pred_train)
    if verbose:
        print("TRAINING")
        print("Confusion Matrix")
        print(confusion_matrix(y_comp, y_pred_train))
        print("Classification report")
        print(classification_report(y_comp, y_pred_train))
        print("Accuracy: ", accuracy_score(y_comp, y_pred_train))
        print("Balanced Accuracy: ", balanced_accuracy_score(y_comp, y_pred_train))
        plot_confusion_matrix(y_comp, y_pred_train)
        print()
        print("TESTING")
        print("Confusion Matrix")
        print(confusion_matrix(y_test, y_pred_test))
        print("Classification report")
        print(classification_report(y_test, y_pred_test))
        print("Accuracy: ", accuracy_score(y_test, y_pred_test))
        print("Balanced Accuracy: ", balanced_accuracy_score(y_test, y_pred_test))
        plot_confusion_matrix(y_test, y_pred_test)
        print()
        print()
    return model

def evaluate_model(model_class, parameters: list, region_name: str, 
                   pos_train: list, neg_train: list, pos_test: list, neg_test: list, 
                   use_positives: bool=True, depth: int=30,
                  verbose: bool=False):
    """Create and evaluate the model

    Args:
        model_class (class): The class of the model
        parameters (list): The list of best hyperparameters
        region_name (str): The name of the region
        pos_train (list): The list of positive samples in the training set
        neg_train (list): The list of negative samples in the training set
        pos_test (list): The list of positive samples in the test set
        neg_test (list): The list of negative samples in the test set
        use_positives (bool, optional): Use positive samples in training (for outlier models). Defaults to True.
        depth (int, optional): The depth to select the regions. Defaults to 30.
        verbose (bool, optional): Print user output. Defaults to False.

    Returns:
        model object: The trained model
    """
    s_list = list()
    s_list.extend(neg_train)
    if use_positives:
        s_list.extend(pos_train)
    df = create_dataframe_of_region(region_name, 
                                     s_list, depth=depth)
    X_train = df.drop(["ihc"], axis=1)
    y_train = df["ihc"]
    s_list = list()
    s_list.extend(neg_train)
    s_list.extend(pos_train)
    df = create_dataframe_of_region(region_name, 
                                     s_list, depth=depth)
    X_comp = df.drop(["ihc"], axis=1)
    y_comp = df["ihc"]
    s_list = list()
    s_list.extend(neg_test)
    s_list.extend(pos_test)
    df = create_dataframe_of_region(region_name, 
                                     s_list, depth=depth)
    X_test = df.drop(["ihc"], axis=1)
    y_test = df["ihc"]
    return evaluate(model_class, parameters, X_train, y_train, X_test, y_test, X_comp, y_comp, verbose=verbose)