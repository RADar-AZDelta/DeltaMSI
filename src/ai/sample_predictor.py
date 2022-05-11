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

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import confusion_matrix

from model.sample import Sample
from ai.training_functions import k_fold
from ai.training_functions import create_dataframe_of_region
from ai.training_functions import extend_parameters
from ai.training_functions import parameter_tuning
from ai.training_functions import evaluate_model

class Sample_Predictor():
    
    def __init__(self, train_pos_list: list, train_neg_list: list, test_pos_list: list, test_neg_list: list, 
                region_name_set: set, depth: int=30, region_min_population_occurance: float=0.75, 
                region_cutoff: float=0.75, region_bacc: float=0.75, verbose: bool=False):
        """Create the predictor (and directly training it)

        Args:
            train_pos_list (list): The list of positive training samples
            train_neg_list (list): The list of negative training samples
            test_pos_list (list): The list of positive test samples
            test_neg_list (list): The list of negative test samples
            region_name_set (set): The names of the regions to use
            depth (int, optional): The depth to use. Defaults to 30.
            region_min_population_occurance (float, optional): Th minimum occurance of a region in the sample population. Defaults to 30.
            region_cutoff (float, optional): The minimum occurance of regions in a sample. Defaults to 0.75.
            region_bacc (float, optional): The minimum balanced accuracy to keep a region. Defaults to 0.75.
            verbose (bool, optional): Print user output. Defaults to False.
        """
        # the minimum depth for a region in a sample
        self.min_depth = depth
        # dict to track region removal
        self.removed_region_dict = dict()
        self.region_freq = dict()
        self.region_bacc = dict()
        # the minimum occurance of a good covered region in the population
        self.region_min_population_occurance = region_min_population_occurance
        # set the training and test sets
        self.train_pos_list = train_pos_list
        self.train_neg_list = train_neg_list
        self.test_pos_list = test_pos_list
        self.test_neg_list = test_neg_list
        # create the k folding
        self.k_fold_dict = k_fold(self.train_pos_list, self.train_neg_list, 3, verbose=verbose)
        # initialise all regions
        self.region_name_set = self.__init_regions(region_name_set, self.region_min_population_occurance, verbose=verbose)
        # filter samples
        self.train_pos_list = self.__filter_sample_list(self.train_pos_list, region_cutoff)
        self.train_neg_list = self.__filter_sample_list(self.train_neg_list, region_cutoff)
        self.test_pos_list = self.__filter_sample_list(self.test_pos_list, region_cutoff)
        self.test_neg_list = self.__filter_sample_list(self.test_neg_list, region_cutoff)
        if verbose:
            print("Samples filtered")
            print("\ttrain: {} pos, {} neg".format(len(self.train_pos_list), len(self.train_neg_list)))
            print("\tvalidate: {} pos, {} neg".format(len(self.test_pos_list), len(self.test_neg_list)))
        # initialise the models
        self.model_dict = dict()
        for region_name in self.region_name_set:
            self.model_dict[region_name] = dict()
        self.__train_logistic_regression(verbose=verbose)
        self.__train_support_vector_machine(verbose=verbose)
        # filter the regions based on the model qualities
        self.final_region_set = self.filter_regions(minimum_region_bacc=region_bacc, verbose=verbose)
        # create the msings baseline
        # self.msings_baseline_dict = self.__create_msings_baseline(verbose=verbose)
        # create the prediction cutoffs
        self.cutoff_dict = self.__create_cutoffs(verbose=verbose)
        
    def get_regions(self) -> set:
        """Returns the regions used in this model

        Returns:
            set: The names of the used regions
        """
        return self.final_region_set
        
    def __init_regions(self, region_name_set: set, min_population_occurance: float, verbose:bool=False) -> set:
        """Filter the regions for occurance in the population

        Args:
            region_name_set (set): The names of the regions to start from
            min_population_occurance (float): The needed frequence of a region in the population
            verbose (bool, optional): Print user output. Defaults to False.

        Returns:
            set: The filtered set of region names
        """
        if verbose:
            print("Initialise the regions")
            print("Start from {} regions".format(len(region_name_set)))
        new_region_set = set()
        for region_name in region_name_set:
            sample_count = 0
            region_count = 0
            for my_list in [self.train_pos_list, self.train_neg_list, self.test_pos_list, self.test_neg_list]:
                for sample in my_list:
                    sample_count += 1
                    if sample.get_region_above_depth(region_name, self.min_depth) is not None:
                        region_count += 1
            self.region_freq[region_name] = region_count/sample_count
            if region_count/sample_count > min_population_occurance:
                new_region_set.add(region_name)
            else:
                if verbose:
                    print("\tRemoving {} ({})".format(region_name, region_count/sample_count))
                self.removed_region_dict[region_name] = "Removed by low population frequency ({})".format(
                    region_count/sample_count)
        if verbose:
            print("Regions initialised: ended with {} regions".format(len(new_region_set)))
            print(new_region_set)
        return new_region_set
    
    def __filter_sample_list(self, sample_list: list, region_perc: float, verbose:bool=False) -> list:
        """Filters the given list for samples with not enough covered regions

        Args:
            sample_list (list): The sample list to filter
            region_perc (float): The percentage of region needed in this sample (above the self.min_depth)
            verbose (bool, optional): Print user output. Defaults to False.

        Returns:
            list: A new list of filtered samples
        """
        new_sample_list = list()
        for sample in sample_list:
            region_count = 0
            for region_name in self.region_name_set:
                if sample.get_region_above_depth(region_name, self.min_depth) is not None:
                    region_count += 1
            if (region_count / len(self.region_name_set)) > region_perc:
                new_sample_list.append(sample)
        return new_sample_list
       
    def __train_logistic_regression(self, verbose: bool=False):
        """Train a Logistic Regression model, and save it (with parameter tuning)

        Args:
            verbose (bool, optional): Print user output. Defaults to False.
        """
        if verbose:
            print("Logistic Regression")

        for region_name in self.region_name_set:
            if verbose:
                print("---------------------")
                print(region_name)
            model_class = LogisticRegression
            parameter_list = [{"solver" : ["liblinear"],
                           "penalty": ["l1", "l2"],
                          "C": [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 100],
                          "class_weight": ["balanced", None]},
                          {"solver" : ["lbfgs"],
                           "penalty": ["l2", "none"],
                          "C": [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 100],
                          "class_weight": ["balanced", None]},
                         ]
            (acc, parameters) = parameter_tuning(model_class, parameter_list, 
                            self.k_fold_dict, region_name, use_positives=True, depth=self.min_depth,
                                                verbose=verbose)
            if verbose:
                print("LogisticRegression")
                print("Best Precision is {}".format(acc))
                print("Best parameters are {}".format(parameters))
            model = evaluate_model(model_class, parameters, region_name, 
                           self.train_pos_list, self.train_neg_list, self.test_pos_list, self.test_neg_list, 
                           use_positives=True, depth=self.min_depth, verbose=verbose)
            self.model_dict[region_name]["LogisticRegression"] = model
    
    def __train_support_vector_machine(self, verbose: bool=False):
        """Train a Support Vector Machine model, and save it (with parameter tuning)

        Args:
            verbose (bool, optional): Print user output. Defaults to False.
        """
        if verbose:
            print("Support Vector Machine")
    
        for region_name in self.region_name_set:
            if verbose:
                print("---------------------")
                print(region_name)
            model_class = SVC
            parameter_list = [{"kernel" : ["linear"],
                            "C": [0.001, 0.01, 0.1, 1, 10, 100],
                            "class_weight": ["balanced", None]},
                          {"kernel" : ["rbf"],
                           "C": [0.001, 0.01, 0.1, 1, 10, 100],
                           "class_weight": ["balanced", None],
                           "gamma" : ["scale", "auto"]},
                         ]
            (acc, parameters) = parameter_tuning(model_class, parameter_list, 
                            self.k_fold_dict, region_name, use_positives=True, depth=self.min_depth,
                                                verbose=verbose)
            if verbose:
                print("SVC")
                print("Best Precision is {}".format(acc))
                print("Best parameters are {}".format(parameters))
            model = evaluate_model(model_class, parameters, region_name, 
                           self.train_pos_list, self.train_neg_list, self.test_pos_list, self.test_neg_list, 
                           use_positives=True, depth=self.min_depth, verbose=verbose)
            self.model_dict[region_name]["SVC"] = model
            
    def filter_regions(self, minimum_region_bacc: float=0.75, verbose: bool=False):
        """Filter the regions after training, only keep those were a model could be 
        created with a minimum balanced accuracy of minimum_region_bacc

        Args:
            minimum_region_bacc (float, optional): The minimum balanced accuracy to keep a region for prediction. Defaults to 0.75.
            verbose (bool, optional): Print user output. Defaults to False.
        """
        if verbose:
            print("Start filtering regions")
            print("Starting from {} regions".format(len(self.region_name_set)))
        new_region_set = set()
        for region_name in self.region_name_set:
            s_list = list()
            s_list.extend(self.test_neg_list)
            s_list.extend(self.test_pos_list)
            df = create_dataframe_of_region(region_name, 
                                             s_list, depth=self.min_depth)
            X_test = df.drop(["ihc"], axis=1)
            y_test = df["ihc"]
            region_model_dict = self.model_dict[region_name]
            pass_count = 0
            bacc_list = list()
            for model_name in region_model_dict:
                model = region_model_dict[model_name]
#                 y_pred_test = predict(model, X_test)
                y_pred_test = model.predict(X_test)
                bacc = balanced_accuracy_score(y_test, y_pred_test)
#                 tn, fp, fn, tp = confusion_matrix(y_test, y_pred_test).ravel()
#                 bacc = tn/(tn+fp)
                bacc_list.append(bacc)
                if bacc > minimum_region_bacc:
                    pass_count += 1
            self.region_bacc[region_name] = bacc_list
            if pass_count == len(region_model_dict):
                new_region_set.add(region_name)
            else:
                self.removed_region_dict[region_name] = "Removed by low prediction accuracy ({})".format(bacc_list)
                if verbose:
                    print("Removing {}, prediction accuracy was {}".format(region_name, bacc_list))
        if verbose:
            print("Ended with {} regions".format(len(new_region_set)))
            print(new_region_set)
        return new_region_set
    
    def region_prediction(self, sample: Sample, region_name: str, verbose: bool=False):
        """Predict the status of the given region for the given sample

        Args:
            sample (Sample): The sample to predict
            region_name (str): The region to predict
            verbose (bool, optional): Print user output. Defaults to False.

        Returns:
            (int, int, int): The results of (log, svm, combined)
        """
        combination = 0
        region_model_dict = self.model_dict[region_name]
        df = create_dataframe_of_region(region_name, [sample], depth=self.min_depth)
        if df.shape[0] == 0:
            return (np.nan, np.nan, np.nan)
        X_test = df.drop(["ihc"], axis=1)
        y_test = df["ihc"]
        log_predict = region_model_dict["LogisticRegression"].predict(X_test)[0]
        svc_predict = region_model_dict["SVC"].predict(X_test)[0]
        return (log_predict, svc_predict, int((log_predict + svc_predict) > 1))
    
    def sample_raw_prediction(self, sample: Sample, verbose: bool=False):
        """Predict the status of the given sample (all regions)

        Args:
            sample (Sample): The sample to predict
            verbose (bool, optional): Print user output. Defaults to False.

        Returns:
            (float, float, float): For (log, svm, combined) the percentage of instable regions
        """
        log_count = 0
        svc_count = 0
        comb_count = 0
        region_count = 0
        for region_name in self.final_region_set:
            (log, svc, comb) = self.region_prediction(sample, region_name, verbose=verbose)
            if str(log) != "nan":
                log_count += log
                svc_count += svc
                comb_count += comb
                region_count +=1
        if region_count < round((len(self.get_regions()) * self.region_min_population_occurance), 2):
            return (np.nan, np.nan, np.nan)
        return (log_count/region_count, svc_count/region_count, comb_count/region_count)
    
    def sample_prediction(self, sample: Sample, verbose: bool=False):
        """Predict the sample (raw and interpretations)

        Args:
            sample (Sample): The sample to predict
            verbose (bool, optional): Print user output. Defaults to False.

        Returns:
            (float, str, float, str, float, str, float, str): The predicted outcome of 
            (log, svm, combined, sample), with both the raw value and the interpretation (MSI, MSS, "" for gray zone)
            log is prediction through Logistic Regression, svm through Support Vector Machine, 
            combined is voting (both must mark a region as instable),
            sample is the combination of the 3 previous interpretations: if 2 out of 3 predicts the same, this is predicted
            (with 0 for MSS, 1 for MSI) otherwise the outcome is "" (gray zone, with value 0.5)
        """
        (log, svc, comb) = self.sample_raw_prediction(sample, verbose=verbose)
        log_p = ""
        svc_p = ""
        comb_p = ""
        sample_p = ""
        if log <= self.cutoff_dict["LogisticRegression"]["Lower"]:
            log_p = "MSS"
        elif log > self.cutoff_dict["LogisticRegression"]["Upper"]:
            log_p = "MSI"
        if svc <= self.cutoff_dict["SVC"]["Lower"]:
            svc_p = "MSS"
        elif svc > self.cutoff_dict["SVC"]["Upper"]:
            svc_p = "MSI"
        if comb <= self.cutoff_dict["Combo"]["Lower"]:
            comb_p = "MSS"
        elif comb > self.cutoff_dict["Combo"]["Upper"]:
            comb_p = "MSI"
        sample = 0.5
        if [log_p, svc_p, comb_p].count("MSS") >= 2:
            sample_p = "MSS"
            sample = 0
        elif [log_p, svc_p, comb_p].count("MSI") >= 2:
            sample_p = "MSI"
            sample = 1
        return (log, log_p, svc, svc_p, comb, comb_p, sample, sample_p)
    
    def __cutoff_optimizer(self, pred_list: list, true_list: list, nr_of_regions: int, cutoff_min: float=0.9):
        """Optimizes the cutoffs for the given predicted list

        Args:
            pred_list (list): The predicted outcomes
            true_list (list): The true (expected) outcomes
            nr_of_regions (int): The number of retions used
            cutoff_min (float, optional): The minimal cutoff of tpr or tnr to set the cutoffs. Defaults to 0.9.

        Returns:
            (float, float, float, float): The precision of the MSS prediction, the lower cutoff, 
                    the precision of the MSI prediction, the upper cutoff
        """
        upper = 0
        upper_val = -1
        lower = 0
        lower_val = -1
        for v in range(nr_of_regions-1, -1, -1):
            my_list = list()
            val = (v/nr_of_regions) + (1/(nr_of_regions*2))
            for i in pred_list:
                if i > val:
                    my_list.append(1)
                else:
                    my_list.append(0)
            tn, fp, fn, tp = confusion_matrix(true_list, my_list).ravel()
#             if lower < tn/(tn+fn) and lower < cutoff_min and tn/(tn+fn) > 0.5:
            if lower < tn/(tn+fn) and lower < cutoff_min:
                lower = tn/(tn+fn)
                lower_val = val
        for v in range(0, nr_of_regions):
            my_list = list()
            val = (v/nr_of_regions) + (1/(nr_of_regions*2))
            for i in pred_list:
                if i > val:
                    my_list.append(1)
                else:
                    my_list.append(0)
            tn, fp, fn, tp = confusion_matrix(true_list, my_list).ravel()
#             if upper < tp/(tp+fp) and upper < cutoff_min and tp/(tp+fp) > 0.5:
            if upper < tp/(tp+fp) and upper < cutoff_min:
                upper = tp/(tp+fp)
                upper_val = val
        if upper_val < lower_val:
            tmp = upper
            tmp_val = upper_val
            upper = lower
            upper_val = lower_val
            lower = tmp
            lower_val = tmp_val
        return (lower, lower_val, upper, upper_val)
    
    def __create_cutoffs(self, verbose: bool=False):
        """Create cutoffs for all modles

        Args:
            verbose (bool, optional): Print user output. Defaults to False.

        Returns:
            dict: a dict with as keys the models, values are dicts with Lower and Upper cutoffs
        """
#         self.test_pos_list, self.test_neg_list
        logres_list = list()
        svc_list = list()
        comb_list = list()
        true_list = list()
        for sample in self.test_pos_list:
            (log, svc, comb) = self.sample_raw_prediction(sample, verbose=verbose)
            logres_list.append(log)
            svc_list.append(svc)
            comb_list.append(comb)
            true_list.append(1)
        for sample in self.test_neg_list:
            (log, svc, comb) = self.sample_raw_prediction(sample, verbose=verbose)
            logres_list.append(log)
            svc_list.append(svc)
            comb_list.append(comb)
            true_list.append(0)
        (_, logres_lower_val, _, logres_upper_val) = self.__cutoff_optimizer(logres_list, true_list, 
                                                                             len(self.final_region_set))
        (_, svc_lower_val, _, svc_upper_val) = self.__cutoff_optimizer(svc_list, true_list, 
                                                                             len(self.final_region_set))
        (_, comb_lower_val, _, comb_upper_val) = self.__cutoff_optimizer(comb_list, true_list, 
                                                                             len(self.final_region_set))
        value_dict = dict()
        value_dict["LogisticRegression"] = dict()
        value_dict["LogisticRegression"]["Lower"] = logres_lower_val
        value_dict["LogisticRegression"]["Upper"] = logres_upper_val
        value_dict["SVC"] = dict()
        value_dict["SVC"]["Lower"] = svc_lower_val
        value_dict["SVC"]["Upper"] = svc_upper_val
        value_dict["Combo"] = dict()
        value_dict["Combo"]["Lower"] = comb_lower_val
        value_dict["Combo"]["Upper"] = comb_upper_val
        return value_dict