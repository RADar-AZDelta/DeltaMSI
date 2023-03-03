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

from deltamsi.io_module.bed_files import create_regions_from_bed
from deltamsi.ai.sample_predictor import Sample_Predictor
from deltamsi.ai.create_profile import get_sample_from_bam
from deltamsi.io_module.ihc_files import update_ihc_status
from deltamsi.ai.training_functions import train_test_split
from deltamsi.io_module.json_files import dict_to_json

from deltamsi.model.sample import Sample 
from deltamsi.model.prediction import Prediction

from joblib import dump as joblibdump 
from joblib import load as joblibload
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

class AIModel():

    def __init__(self, bed_regions: list, regions: list, predictor: Sample_Predictor, flanking: int, minimum_mapping_quality: int, depth: int):
        """Create a new AI model (this object is saved and reused)

        Args:
            bed_regions (list): A list of the bed regions
            regions (list): A list of the regions
            predictor (Sample_Predictor): A trained predictor object
            flanking (int): The number of bases to use in the flanking
            minimum_mapping_quality (int): The minimum quality for the mapping
            depth (int): The minimum depth of a region
        """
        self._predictor = predictor
        self._bed_regions = bed_regions
        self._regions = regions
        self._flanking = flanking
        self._minimum_mapping_quality = minimum_mapping_quality
        self._depth = depth
        # create the predictions of the train and validation sets
        self._pos_train_predictions = list()
        for sample in self._predictor.train_pos_list:
            self._pos_train_predictions.append(self.predict_sample(sample))
        self._neg_train_predictions = list()
        for sample in self._predictor.train_neg_list:
            self._neg_train_predictions.append(self.predict_sample(sample))
        self._pos_val_predictions = list()
        for sample in self._predictor.test_pos_list:
            self._pos_val_predictions.append(self.predict_sample(sample))
        self._neg_val_predictions = list()
        for sample in self._predictor.test_neg_list:
            self._neg_val_predictions.append(self.predict_sample(sample))

    @property 
    def regions(self) -> set:
        """Get all regions

        Returns:
            set: A set of regions
        """
        return self._regions

    @property
    def bed_regions(self) -> list:
        """Get all bed regions, used to read a bam file

        Returns:
            list: The bed regions
        """
        return self._bed_regions

    def get_bed_region(self, name: str) -> 'Region':
        """Get a bed regions with the given name

        Args:
            name (str): The name of the region

        Returns:
            Region: The bed region with that name
        """
        for region in self.bed_regions:
            if region.name == name:
                return region
        return None

    @property 
    def flanking(self) -> int:
        """Get the number of bases that a region must flank

        Returns:
            int: The number of flanking bases
        """
        return self._flanking

    @property 
    def minimum_mapping_quality(self) -> int:
        """Get the minimum mapping quality for a read

        Returns:
            int: The minimum mapping quality
        """
        return self._minimum_mapping_quality

    @property 
    def depth(self) -> int:
        """Get the minimum depth of a region

        Returns:
            int: The minimum depth
        """
        return self._depth

    def predict_sample(self, sample: Sample) -> Prediction:
        """Get the prediction of the given sample

        Args:
            sample (Sample): The sample to predict

        Returns:
            Prediction: The complete prediction of the sample
        """
        # predict (sample + regions)
        prediction = Prediction(sample)
        (log, log_p, svc, svc_p, vot, vot_p, sam, sam_p) = self._predictor.sample_prediction(sample)
        prediction.set_prediction(log, log_p, svc, svc_p, vot, vot_p, sam, sam_p)
        for region_name in self.regions:
            (log, svc, comb) = self._predictor.region_prediction(sample, region_name)
            prediction.add_region(region_name, log, svc, comb)
        return prediction

    def predict_output_creation(self, prediction: Prediction, output_dir: str):
        """Create all output for a given prediction
                This creates a result.tsv with the results of the sample, and per region
                This creates distribution graphs per region
                This creates a prediction plot showing the values of this sample vs the train/validation set and the cutoffs
                This creates a result.json, which can be used in other applications

        Args:
            prediction (Prediction): The prediction to generate the output
            output_dir (str): The output directory
        """
        # create result file (sample prediction)
        json_dict = dict()
        for col in ["sample", "regions", "cutoffplot"]:
            json_dict[col] = dict()
        (log, svm, vot, sam) = prediction.get_prediction_values()
        (log_p, svm_p, vot_p, sam_p) = prediction.get_predictions()
        json_dict["sample"]["LogisticRegression"] = dict()
        json_dict["sample"]["LogisticRegression"]["raw"] = log
        json_dict["sample"]["LogisticRegression"]["predicted"] = log_p
        json_dict["sample"]["SupportVectorMachine"] = dict()
        json_dict["sample"]["SupportVectorMachine"]["raw"] = svm
        json_dict["sample"]["SupportVectorMachine"]["predicted"] = svm_p
        json_dict["sample"]["VotingModel"] = dict()
        json_dict["sample"]["VotingModel"]["raw"] = vot
        json_dict["sample"]["VotingModel"]["predicted"] = vot_p
        json_dict["sample"]["SampleModel"] = dict()
        json_dict["sample"]["SampleModel"]["raw"] = sam
        json_dict["sample"]["SampleModel"]["predicted"] = sam_p
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open("{}/{}.result.tsv".format(output_dir, prediction.sample.name), 'w+') as f:
            f.write("Sample Name\t{}\n".format(prediction.sample.name))
            f.write("Logistic Regression\t{:.2f}\t{}\n".format(log, log_p))
            f.write("SVM\t{:.2f}\t{}\n".format(svm, svm_p))
            f.write("Voting\t{:.2f}\t{}\n".format(vot, vot_p))
            f.write("Sample Conclusion\t{:.2f}\t{}\n".format(sam, sam_p))
            f.write("-----\t-----\t-----\n")
            f.write("Region Name\tLogistic Regression\tSVM\tVoting\n")
            f.write("-----\t-----\t-----\t-----\n")
            for region_name in sorted(list(self.regions)):
                (log, svm, vot) = prediction.get_region(region_name)
                f.write("{}\t{}\t{}\t{}\n".format(region_name, log, svm, vot))
        # plot region vs neg regions
        for region_name in self.regions:
            json_dict["regions"][region_name] = dict()
            region = self.get_bed_region(region_name)
            json_dict["regions"][region_name]["name"] = region_name
            json_dict["regions"][region_name]["chr"] = region.chr
            json_dict["regions"][region_name]["start"] = region.start
            json_dict["regions"][region_name]["end"] = region.end
            json_dict["regions"][region_name]["predicted"] = dict()
            json_dict["regions"][region_name]["length_graph"] = list()
            json_dict["regions"][region_name]["length_graph_trainset"] = list()
            plt.figure(figsize=(10,10))
            for train_pred in self._neg_train_predictions:
                region = train_pred.sample.get_region(region_name)
                if region is not None:
                    plt.plot(range(0, region.region_length+1), region.get_graph(flex_cutoff=True), "#808080")
                    json_dict["regions"][region_name]["length_graph_trainset"].append(region.get_graph(flex_cutoff=True))
            (log, svm, vot) = prediction.get_region(region_name)
            region = prediction.sample.get_region(region_name)
            json_dict["regions"][region_name]["length_graph"] = region.get_graph(flex_cutoff=True)
            if str(log) == "nan":
                json_dict["regions"][region_name]["predicted"]["LogisticRegression"] = None
            else:
                json_dict["regions"][region_name]["predicted"]["LogisticRegression"] = int(log)
            if str(svm) == "nan":
                json_dict["regions"][region_name]["predicted"]["SupportVectorMachine"] = None
            else:
                json_dict["regions"][region_name]["predicted"]["SupportVectorMachine"] = int(svm)
            if str(vot) == "nan":
                json_dict["regions"][region_name]["predicted"]["VotingModel"] = None 
            else:
                json_dict["regions"][region_name]["predicted"]["VotingModel"] = int(vot)
            if sum([log, svm, vot]) >= 2:
                # color pos (red)
                plt.plot(range(0, region.region_length+1), region.get_graph(flex_cutoff=True), "#e60000")
            elif sum([log, svm, vot]) == 0:
                # color neg (green)
                plt.plot(range(0, region.region_length+1), region.get_graph(flex_cutoff=True), "#32d232")
            else:
                # color gray zone (yellow)
                plt.plot(range(0, region.region_length+1), region.get_graph(flex_cutoff=True), "#FFFF00")
            plt.title(region_name)
            plt.savefig("{}/{}.distribution.png".format(output_dir, region_name))
        # plot sample vs training in quadrants
        log_neg = list()
        log_pos = list()
        svc_neg = list()
        svc_pos = list()
        vot_neg = list()
        vot_pos = list()
        for prediction_list in [self._pos_train_predictions, self._neg_train_predictions,
                    self._pos_val_predictions, self._neg_val_predictions]:
            for pred in prediction_list:
                (log, svc, vot, sam) = pred.get_prediction_values()
                if pred.sample.ihc:
                    log_pos.append(log)
                    svc_pos.append(svc)
                    vot_pos.append(vot)
                else:
                    log_neg.append(log)
                    svc_neg.append(svc)
                    vot_neg.append(vot)
        (log, svm, vot, sam) = prediction.get_prediction_values()
        sample_log = [log]
        sample_svm = [svm]
        sample_vot = [vot]
        # fill json
        json_dict["cutoffplot"]["LogisticRegression"] = dict()
        json_dict["cutoffplot"]["LogisticRegression"]["lower_cutoff"] = self._predictor.cutoff_dict["LogisticRegression"]["Lower"]
        json_dict["cutoffplot"]["LogisticRegression"]["upper_cutoff"] = self._predictor.cutoff_dict["LogisticRegression"]["Upper"]
        json_dict["cutoffplot"]["LogisticRegression"]["pos_samples"] = log_pos
        json_dict["cutoffplot"]["LogisticRegression"]["neg_samples"] = log_neg
        json_dict["cutoffplot"]["LogisticRegression"]["sample"] = sample_log
        json_dict["cutoffplot"]["SupportVectorMachine"] = dict()
        json_dict["cutoffplot"]["SupportVectorMachine"]["lower_cutoff"] = self._predictor.cutoff_dict["SVC"]["Lower"]
        json_dict["cutoffplot"]["SupportVectorMachine"]["upper_cutoff"] = self._predictor.cutoff_dict["SVC"]["Upper"]
        json_dict["cutoffplot"]["SupportVectorMachine"]["pos_samples"] = svc_pos
        json_dict["cutoffplot"]["SupportVectorMachine"]["neg_samples"] = svc_neg
        json_dict["cutoffplot"]["SupportVectorMachine"]["sample"] = sample_svm
        json_dict["cutoffplot"]["VotingModel"] = dict()
        json_dict["cutoffplot"]["VotingModel"]["lower_cutoff"] = self._predictor.cutoff_dict["Combo"]["Lower"]
        json_dict["cutoffplot"]["VotingModel"]["upper_cutoff"] = self._predictor.cutoff_dict["Combo"]["Upper"]
        json_dict["cutoffplot"]["VotingModel"]["pos_samples"] = vot_pos
        json_dict["cutoffplot"]["VotingModel"]["neg_samples"] = vot_neg
        json_dict["cutoffplot"]["VotingModel"]["sample"] = sample_vot
        dict_to_json("{}/result.json".format(output_dir), json_dict)
        # Logistic Regression vs Suport Vector Machine
        plt.figure(figsize=(10,10))
        plt.plot(([self._predictor.cutoff_dict["LogisticRegression"]["Lower"]]*2), [0,1], 
                label="LogResLower", color="black", linestyle="dotted")
        plt.plot(([self._predictor.cutoff_dict["LogisticRegression"]["Upper"]]*2), [0,1], 
                label="LogResUpper", color="black", linestyle="dotted")
        plt.plot([0,1], ([self._predictor.cutoff_dict["SVC"]["Lower"]]*2), 
                label="SVCLower", color="brown", linestyle="dotted")
        plt.plot([0,1], ([self._predictor.cutoff_dict["SVC"]["Upper"]]*2), 
                label="SVCUpper", color="brown", linestyle="dotted")
        plt.scatter(log_neg, svc_neg, label="Negatives", color="blue")
        plt.scatter(log_pos, svc_pos, label="Positives", color="green")
        plt.scatter(sample_log, sample_svm, label="Sample", color="yellow")
        plt.xlabel('Logistic Regression')
        plt.ylabel('Support Vector Machine')
        plt.legend()
        plt.title("Prediction of {}".format(prediction.sample.name))
        plt.savefig("{}/prediction_plot_LogRes_vs_SVM.png".format(output_dir))
        # Logistic Regression vs Voting
        plt.figure(figsize=(10,10))
        plt.plot(([self._predictor.cutoff_dict["LogisticRegression"]["Lower"]]*2), [0,1], 
                label="LogResLower", color="black", linestyle="dotted")
        plt.plot(([self._predictor.cutoff_dict["LogisticRegression"]["Upper"]]*2), [0,1], 
                label="LogResUpper", color="black", linestyle="dotted")
        plt.plot([0,1], ([self._predictor.cutoff_dict["Combo"]["Lower"]]*2), 
                label="VotingLower", color="brown", linestyle="dotted")
        plt.plot([0,1], ([self._predictor.cutoff_dict["Combo"]["Upper"]]*2), 
                label="VotingUpper", color="brown", linestyle="dotted")
        plt.scatter(log_neg, vot_neg, label="Negatives", color="blue")
        plt.scatter(log_pos, vot_pos, label="Positives", color="green")
        plt.scatter(sample_log, sample_svm, label="Sample", color="yellow")
        plt.xlabel('Logistic Regression')
        plt.ylabel('Voting Model')
        plt.legend()
        plt.title("Prediction of {}".format(prediction.sample.name))
        plt.savefig("{}/prediction_plot_LogRes_vs_Voting.png".format(output_dir))
        # Voting vs Suport Vector Machine
        plt.figure(figsize=(10,10))
        plt.plot(([self._predictor.cutoff_dict["Combo"]["Lower"]]*2), [0,1], 
                label="LogResLower", color="black", linestyle="dotted")
        plt.plot(([self._predictor.cutoff_dict["Combo"]["Upper"]]*2), [0,1], 
                label="LogResUpper", color="black", linestyle="dotted")
        plt.plot([0,1], ([self._predictor.cutoff_dict["SVC"]["Lower"]]*2), 
                label="VotingLower", color="brown", linestyle="dotted")
        plt.plot([0,1], ([self._predictor.cutoff_dict["SVC"]["Upper"]]*2), 
                label="VotingUpper", color="brown", linestyle="dotted")
        plt.scatter(vot_neg, svc_neg, label="Negatives", color="blue")
        plt.scatter(vot_pos, svc_pos, label="Positives", color="green")
        plt.scatter(sample_log, sample_svm, label="Sample", color="yellow")
        plt.xlabel('Voting Model')
        plt.ylabel('Support Vector Machine')
        plt.legend()
        plt.title("Prediction of {}".format(prediction.sample.name))
        plt.savefig("{}/prediction_plot_Voting_vs_SVM.png".format(output_dir))
                    
    def predict_output_creation_multi(self, prediction_list: list, output_dir: str, add_ihc_info: bool=False, verbose: bool=False):
        """This create the output for multiple samples, by generating the output per samples, and generating a result_overview
                combining the outcome of all samples.

        Args:
            prediction_list (list): The list of predictions to generate the output
            output_dir (str): The output directory for all output
            add_ihc_info (bool, optional): Use ihc infromation in the output (for train and evaluation). Defaults to False.
            verbose (bool, optional): Print user information. Defaults to False.
        """ 
        # predict output for each prediciont (seperate output dir)
        for prediction in tqdm(prediction_list, desc="Creating sample output"):
            self.predict_output_creation(prediction, "{}/{}/".format(output_dir, prediction.sample.name))
        # combination of predictions in 1 excel/csv file
        my_dict = dict()
        for col in ["Sample_name", "LogisticRegression_value", "LogisticRegression_prediction",
                    "SVM_value", "SVM_prediction", "Voting_value", "Voting_prediction",
                    "SampleConclusion_value", "SampleConclussion_prediction"]:
            my_dict[col] = list()
        if add_ihc_info:
            my_dict["IHC"] = list()
            my_dict["SAME"] = list()
        my_dict[""] = list()
        for region_name in sorted(list(self.regions)):
            my_dict["{}_LogisticRegression".format(region_name)] = list()
            my_dict["{}_SVM".format(region_name)] = list()
            my_dict["{}_Voting".format(region_name)] = list()
        for prediction in prediction_list:
            if verbose:
                print("\tCreating the output of {}".format(prediction.sample.name))
            my_dict["Sample_name"].append(prediction.sample.name)
            (log, svm, vot, sam) = prediction.get_prediction_values()
            (log_p, svm_p, vot_p, sam_p) = prediction.get_predictions()
            my_dict["LogisticRegression_value"].append(log)
            my_dict["LogisticRegression_prediction"].append(log_p)
            my_dict["SVM_value"].append(svm)
            my_dict["SVM_prediction"].append(svm_p)
            my_dict["Voting_value"].append(vot)
            my_dict["Voting_prediction"].append(vot_p)
            my_dict["SampleConclusion_value"].append(sam)
            my_dict["SampleConclussion_prediction"].append(sam_p)
            if add_ihc_info:
                if prediction.sample.ihc is None:
                    my_dict["IHC"].append(None)
                    my_dict["SAME"].append(None)
                else:
                    if prediction.sample.ihc:
                        my_dict["IHC"].append("MSI")
                    else:
                        my_dict["IHC"].append("MSS")
                    same = (prediction.sample.ihc and sam_p == "MSI") or (not prediction.sample.ihc and sam_p == "MSS")
                    my_dict["SAME"].append(same)
            my_dict[""].append("")
            for region_name in self.regions:
                (log, svm, vot) = prediction.get_region(region_name)
                my_dict["{}_LogisticRegression".format(region_name)].append(log)
                my_dict["{}_SVM".format(region_name)].append(svm)
                my_dict["{}_Voting".format(region_name)].append(vot)
        complete_df = pd.DataFrame.from_dict(my_dict)
        complete_df = complete_df.transpose()
        header = complete_df.iloc[0]
        complete_df = complete_df[1:]
        complete_df.columns = header
        complete_df.to_excel("{}/result_overview.xlsx".format(output_dir), index=True, header=True)
        complete_df.to_csv("{}/result_overview.tsv".format(output_dir), sep='\t', index=True, header=True)

    def create_training_info(self, output_dir: str):
        """Generate trainings information about the model.
                This generates cutoff matrices to show the relation between the samples and the cutoffs
                This generates a text file with the used cutoffs
                This generates different regions files with information about the filtering and removal
                This generates a roc curve, based on the validation samples

        Args:
            output_dir (str): The output directory
        """
        # Create regions selection files
        with open("{}/regions.txt".format(output_dir), 'w+') as f:
            f.write("Used regions:\n")
            for region_name in sorted(list(self.regions)):
                f.write("\t{}\n".format(region_name))
            f.write("\n\nRemoved regions:\n")
            for region_name in self._predictor.removed_region_dict:
                f.write("\t{}\t{}\n".format(region_name, self._predictor.removed_region_dict[region_name]))
        # Create excel with region information
        my_dict = dict()
        for col in ["chr", "start", "end", "name", "population_frequency", "prediction_balanced_accuracy_LogRes",
            "prediction_balanced_accuracy_SVM", "Used"]:
            my_dict[col] = list()
        for region in self._bed_regions:
            my_dict["name"].append(region.name)
            my_dict["chr"].append(region.chr)
            my_dict["start"].append(region.start)
            my_dict["end"].append(region.end)
            my_dict["population_frequency"].append(self._predictor.region_freq.get(region.name))
            if self._predictor.region_bacc.get(region.name) is None:
                my_dict["prediction_balanced_accuracy_LogRes"].append(None)
                my_dict["prediction_balanced_accuracy_SVM"].append(None)
            else:
                my_dict["prediction_balanced_accuracy_LogRes"].append(self._predictor.region_bacc.get(region.name)[0])
                my_dict["prediction_balanced_accuracy_SVM"].append(self._predictor.region_bacc.get(region.name)[1])
            my_dict["Used"].append(region.name in self.regions)
        region_df = pd.DataFrame.from_dict(my_dict)
        # region_df = region_df.transpose()
        # header = region_df.iloc[0]
        # region_df = region_df[1:]
        # region_df.columns = header
        region_df.to_excel("{}/regions.xlsx".format(output_dir), index=False, header=True)
        region_df.to_csv("{}/regions.tsv".format(output_dir), sep='\t', index=False, header=False)
        # create cutoff file
        with open("{}/cutoffs.txt".format(output_dir), 'w+') as f:
            f.write("Used cutoffs:\n")
            f.write("Logistic Regression: [{:.4f}; {:.4f}]\n".format(self._predictor.cutoff_dict["LogisticRegression"]["Lower"],
                        self._predictor.cutoff_dict["LogisticRegression"]["Upper"]))
            f.write("SVM: [{:.4f}; {:.4f}]\n".format(self._predictor.cutoff_dict["SVC"]["Lower"],
                        self._predictor.cutoff_dict["SVC"]["Upper"]))
            f.write("Voting: [{:.4f}; {:.4f}]\n".format(self._predictor.cutoff_dict["Combo"]["Lower"],
                        self._predictor.cutoff_dict["Combo"]["Upper"]))
        # plot cutoff matix
        log_neg = list()
        log_pos = list()
        svc_neg = list()
        svc_pos = list()
        vot_neg = list()
        vot_pos = list()
        for prediction_list in [self._pos_train_predictions, self._neg_train_predictions,
                    self._pos_val_predictions, self._neg_val_predictions]:
            for pred in prediction_list:
                (log, svc, vot, sam) = pred.get_prediction_values()
                if pred.sample.ihc:
                    log_pos.append(log)
                    svc_pos.append(svc)
                    vot_pos.append(vot)
                else:
                    log_neg.append(log)
                    svc_neg.append(svc)
                    vot_neg.append(vot)
        # Cutoff matrix LogRes vs SVM
        plt.figure(figsize=(10,10))
        plt.plot(([self._predictor.cutoff_dict["LogisticRegression"]["Lower"]]*2), [0,1], 
                label="LogResLower", color="black", linestyle="dotted")
        plt.plot(([self._predictor.cutoff_dict["LogisticRegression"]["Upper"]]*2), [0,1], 
                label="LogResUpper", color="black", linestyle="dotted")
        plt.plot([0,1], ([self._predictor.cutoff_dict["SVC"]["Lower"]]*2), 
                label="SVCLower", color="brown", linestyle="dotted")
        plt.plot([0,1], ([self._predictor.cutoff_dict["SVC"]["Upper"]]*2), 
                label="SVCUpper", color="brown", linestyle="dotted")
        plt.scatter(log_neg, svc_neg, label="Negatives", color="blue")
        plt.scatter(log_pos, svc_pos, label="Positives", color="green")
        plt.xlabel('Logistic Regression')
        plt.ylabel('Support Vector Machine')
        plt.legend()
        plt.savefig("{}/cutoff_matrix_LogRes_vs_SVM.png".format(output_dir))
        # Cutoff matrix LogRes vs Vot
        plt.figure(figsize=(10,10))
        plt.plot(([self._predictor.cutoff_dict["LogisticRegression"]["Lower"]]*2), [0,1], 
                label="LogResLower", color="black", linestyle="dotted")
        plt.plot(([self._predictor.cutoff_dict["LogisticRegression"]["Upper"]]*2), [0,1], 
                label="LogResUpper", color="black", linestyle="dotted")
        plt.plot([0,1], ([self._predictor.cutoff_dict["Combo"]["Lower"]]*2), 
                label="VotingLower", color="brown", linestyle="dotted")
        plt.plot([0,1], ([self._predictor.cutoff_dict["Combo"]["Upper"]]*2), 
                label="VotingUpper", color="brown", linestyle="dotted")
        plt.scatter(log_neg, vot_neg, label="Negatives", color="blue")
        plt.scatter(log_pos, vot_pos, label="Positives", color="green")
        plt.xlabel('Logistic Regression')
        plt.ylabel('Voting Model')
        plt.legend()
        plt.savefig("{}/cutoff_matrix_LogRes_vs_Voting.png".format(output_dir))
        # Cutoff matrix Vot vs SVM
        plt.figure(figsize=(10,10))
        plt.plot(([self._predictor.cutoff_dict["Combo"]["Lower"]]*2), [0,1], 
                label="VotingLower", color="black", linestyle="dotted")
        plt.plot(([self._predictor.cutoff_dict["Combo"]["Upper"]]*2), [0,1], 
                label="VotingUpper", color="black", linestyle="dotted")
        plt.plot([0,1], ([self._predictor.cutoff_dict["SVC"]["Lower"]]*2), 
                label="SVCLower", color="brown", linestyle="dotted")
        plt.plot([0,1], ([self._predictor.cutoff_dict["SVC"]["Upper"]]*2), 
                label="SVCUpper", color="brown", linestyle="dotted")
        plt.scatter(vot_neg, svc_neg, label="Negatives", color="blue")
        plt.scatter(vot_pos, svc_pos, label="Positives", color="green")
        plt.xlabel('Logistic Regression')
        plt.ylabel('Support Vector Machine')
        plt.legend()
        plt.savefig("{}/cutoff_matrix_Voting_vs_SVM.png".format(output_dir))
        # plot ROC curve
        log_list = list()
        svc_list = list()
        comb_list = list()
        true_list = list()
        for pred_list in [self._pos_train_predictions, self._neg_train_predictions,
                    self._pos_val_predictions, self._neg_val_predictions]:
            for pred in pred_list:
                true_list.append(pred.sample.ihc)
                (log, svc, comb, _) = pred.get_prediction_values()
                (log_p, svc_p, comb_p, _) = pred.get_predictions()
                # (log, log_p, svc, svc_p, comb, comb_p, _, _) = self._predictor.sample_prediction(sample)
                if str(log) == 'nan':
                    log = 0
                if str(svc) == 'nan':
                    svc = 0
                if str(comb) == 'nan':
                    comb = 0
                log_list.append(log)
                svc_list.append(svc)
                comb_list.append(comb)
                
        ns_prob = [0 for _ in range(len(true_list))]
        ns_fpr, ns_tpr, _ = roc_curve(true_list, ns_prob)
        logres_fpr, logres_tpr, _ = roc_curve(true_list, log_list)
        svc_fpr, svc_tpr, _ = roc_curve(true_list, svc_list)
        comb_fpr, comb_tpr, _ = roc_curve(true_list, comb_list)

        plt.figure(figsize=(20,20))
        plt.plot(ns_fpr, ns_tpr, label="No Prediction ({:.2f})".format(roc_auc_score(true_list, ns_prob)))
        plt.plot(logres_fpr, logres_tpr, label="LogisticRegression ({:.2f})".format(roc_auc_score(true_list, log_list)))
        plt.plot(svc_fpr, svc_tpr, label="SVC ({:.2f})".format(roc_auc_score(true_list, svc_list)))
        plt.plot(comb_fpr, comb_tpr, label="RegionCombination ({:.2f})".format(roc_auc_score(true_list, comb_list)))
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.savefig("{}/roc.png".format(output_dir))

    @staticmethod
    def create_model(bed_file: str, bam_list: list, ihc_file: str, output_dir: str, 
                    flanking: int, minimum_mapping_quality: int, depth: int, verbose: bool=False):
        """This static method creates a model, and saves it to a file.
                For all samples the prediction is generated.
                The trainings information is generated for the user

        Args:
            bed_file (str): The path to the bed file to use
            bam_list (list): A list with all paths to all bam files
            ihc_file (str): The path to the csv/tsv with the ihc information
            output_dir (str): The path to the output directory
            flanking (int): The number of bases to flank regions
            minimum_mapping_quality (int): The minimum mapping quality to use in parsing the regions
            depth (int): The minimum depth to use per region
            verbose (bool, optional): Print user information. Defaults to False.

        Returns:
            AIModel: The build and ready to use AI model
        """
        if verbose:
            print("Start reading the resources")
            print("Reading the bed file")
        # read all regions from bed
        regions = create_regions_from_bed(bed_file)
        region_name_set = set()
        for region in regions:
            region_name_set.add(region.name)
        # create profile of all bams + create Samples
        if verbose:
            print("Reading the bam files")
        sample_dict = dict()
        for bam_file in tqdm(bam_list, desc="Bamfiles"):
            sample = get_sample_from_bam(bam_file, regions, flanking, minimum_mapping_quality)
            sample_dict[sample.name] = sample
        # read references
        if verbose:
            print("Start reading the ihc file")
        sample_dict = update_ihc_status(ihc_file, sample_dict)
        # do train-test split
        ihc_list = list()
        for sample_name in sample_dict:
            sample = sample_dict[sample_name]
            if sample.ihc is not None:
                ihc_list.append(sample)
        if len(ihc_list) < 10:
            import sys
            print("Did not find enough samples to start the training")
            sys.exit()
        (pos_train, neg_train, pos_val, neg_val) = train_test_split(ihc_list, test_perc=0.333, verbose=verbose)
        # train model
        if verbose:
            print("Start training the models")
        predictor = Sample_Predictor(pos_train, neg_train, pos_val, neg_val, region_name_set, depth, 0.75, region_cutoff=0.75, region_bacc=0.6, verbose=verbose)
        if verbose:
            print("Predicting the input samples, and generate training info")
        aimodel = AIModel(regions, predictor.get_regions(), predictor, flanking, minimum_mapping_quality, depth)
        # create output for the model
        prediction_list = list()
        prediction_list.extend(aimodel._pos_train_predictions)
        prediction_list.extend(aimodel._neg_train_predictions)
        prediction_list.extend(aimodel._pos_val_predictions)
        prediction_list.extend(aimodel._neg_val_predictions)
        if verbose:
            print("Start generating result files of {} samples".format(len(prediction_list)))
        aimodel.predict_output_creation_multi(prediction_list, output_dir, add_ihc_info=True, verbose=verbose)
        if verbose:
            print("\tGenerate model information")
        aimodel.create_training_info(output_dir)
        # save aimodel
        if verbose:
            print("Save the model")
        joblibdump(aimodel, "{}/aimodel.deltamsi".format(output_dir))
        return aimodel

    @staticmethod
    def load_model(model_dir: str):
        """Read the given model, and return

        Args:
            model_dir (str): The path to the model, or the directory of the model

        Returns:
            AIModel: The loaded AI model
        """
        if model_dir.endswith("/aimodel.deltamsi"):
            model_path = model_dir
        else:
            model_path = "{}/aimodel.deltamsi".format(model_dir)
        aimodel = joblibload(model_path)
        return aimodel

    @staticmethod
    def predict(model_dir: str, bam_list: list, output_dir: str, verbose: bool=False):
        """Load the model, predict the given samples and generate all output

        Args:
            model_dir (str): The path to the model
            bam_list (list): A list of all bam files
            output_dir (str): The path to the output directory
            verbose (bool, optional): Generate user information. Defaults to False.
        """
        if verbose:
            print("Loading the model")
        aimodel = AIModel.load_model(model_dir)
        prediction_list = list()
        if verbose:
            print("Start the predictions")
        for bam_file in tqdm(bam_list, desc="Reading and Predicting:"):
            sample = get_sample_from_bam(bam_file, aimodel.bed_regions, aimodel.flanking, aimodel.minimum_mapping_quality)
            prediction_list.append(aimodel.predict_sample(sample))
        if verbose:
            print("Create the sample output")
        if len(prediction_list) == 1:
            prediction = prediction_list[0]
            aimodel.predict_output_creation(prediction, output_dir)
        elif len(prediction_list) > 1:
            aimodel.predict_output_creation_multi(prediction_list, output_dir, verbose=verbose)

    @staticmethod
    def evaluate(model_dir: str, bam_list: list, ihc_file: str, output_dir: str, verbose: bool=False):
        """Evaluates the model (same as prediction, but calculates ROC and evalutation metrics)

        Args:
            model_dir (str): The path to the model
            bam_list (list): A list of all bam files
            ihc_file (str): The path to the file with the IHC information
            output_dir (str): The output directory
            verbose (bool, optional): Generate user information. Defaults to False.
        """
        if verbose:
            print("Loading the model")
        aimodel = AIModel.load_model(model_dir)
        prediction_list = list()
        if verbose:
            print("Reading the bam files")
        sample_dict = dict()
        for bam_file in tqdm(bam_list, desc="Bamfiles"):
            sample = get_sample_from_bam(bam_file, aimodel.bed_regions, aimodel.flanking, aimodel.minimum_mapping_quality)
            sample_dict[sample.name] = sample
        # read references
        if verbose:
            print("Start reading the ihc file")
        sample_dict = update_ihc_status(ihc_file, sample_dict)
        if verbose:
            print("Start the predictions")
        for sample_name in tqdm(sample_dict, desc="Predicting"):
            prediction_list.append(aimodel.predict_sample(sample_dict[sample_name]))
        if verbose:
            print("Create the sample output")
        aimodel.predict_output_creation_multi(prediction_list, output_dir, add_ihc_info=True, verbose=verbose)
        # create evaluation overview
        with open("{}/evaluation.txt".format(output_dir), 'w+') as f:
            f.write("# Sample Prediction\n")
            f.write("Confusion matrix:\n")
            pos_pos = 0
            pos_gray = 0
            pos_neg = 0
            neg_pos = 0
            neg_gray = 0
            neg_neg = 0
            for prediction in prediction_list:
                (log_p, svc_p, comb_p, sam_p) = prediction.get_predictions()
                if prediction.sample.ihc is not None:
                    if prediction.sample.ihc:
                        if sam_p == "MSI":
                            pos_pos += 1
                        elif sam_p == "MSS":
                            pos_neg += 1
                        else:
                            pos_gray += 1
                    else:
                        if sam_p == "MSI":
                            neg_pos += 1
                        elif sam_p == "MSS":
                            neg_neg += 1
                        else:
                            neg_gray += 1
            f.write("\tMSI\tGray\tMSS\n")
            f.write("dMMR\t{}\t{}\t{}\n".format(pos_pos, pos_gray, pos_neg))
            f.write("pMMR\t{}\t{}\t{}\n".format(neg_pos, neg_gray, neg_neg))
            f.write("\n")
            f.write("Accuracy: {}\n".format(sum([pos_pos, neg_neg]) / sum([pos_pos, pos_gray, pos_neg, neg_pos, neg_gray, neg_neg])))
            bacc = (pos_pos / sum([pos_pos, pos_gray, pos_neg]) + neg_neg / sum([neg_neg, neg_gray, neg_pos])) / 2
            f.write("Balanced Accuracy: {}\n".format(bacc))
            f.write("Precision MSI: {}\n".format(pos_pos / sum([pos_pos, neg_pos])))
            f.write("Recall MSI: {}\n".format(pos_pos / sum([pos_pos, pos_gray, pos_neg])))
            f.write("Precision MSS: {}\n".format(neg_neg / sum([neg_neg, pos_neg])))
            f.write("Recall MSS: {}\n".format(neg_neg / sum([neg_neg, neg_gray, neg_pos])))
            f.write("\n")
            
            f.write("# Logistic Regression Prediction\n")
            f.write("Confusion matrix:\n")
            pos_pos = 0
            pos_gray = 0
            pos_neg = 0
            neg_pos = 0
            neg_gray = 0
            neg_neg = 0
            for prediction in prediction_list:
                (log_p, svc_p, comb_p, sam_p) = prediction.get_predictions()
                if prediction.sample.ihc:
                    if log_p == "MSI":
                        pos_pos += 1
                    elif log_p == "MSS":
                        pos_neg += 1
                    else:
                        pos_gray += 1
                else:
                    if log_p == "MSI":
                        neg_pos += 1
                    elif log_p == "MSS":
                        neg_neg += 1
                    else:
                        neg_gray += 1
            f.write("\tMSI\tGray\tMSS\n")
            f.write("dMMR\t{}\t{}\t{}\n".format(pos_pos, pos_gray, pos_neg))
            f.write("pMMR\t{}\t{}\t{}\n".format(neg_pos, neg_gray, neg_neg))
            f.write("\n")
            f.write("Accuracy: {}\n".format(sum([pos_pos, neg_neg]) / sum([pos_pos, pos_gray, pos_neg, neg_pos, neg_gray, neg_neg])))
            bacc = (pos_pos / sum([pos_pos, pos_gray, pos_neg]) + neg_neg / sum([neg_neg, neg_gray, neg_pos])) / 2
            f.write("Balanced Accuracy: {}\n".format(bacc))
            f.write("Precision MSI: {}\n".format(pos_pos / sum([pos_pos, neg_pos])))
            f.write("Recall MSI: {}\n".format(pos_pos / sum([pos_pos, pos_gray, pos_neg])))
            f.write("Precision MSS: {}\n".format(neg_neg / sum([neg_neg, pos_neg])))
            f.write("Recall MSS: {}\n".format(neg_neg / sum([neg_neg, neg_gray, neg_pos])))
            f.write("\n")
            
            f.write("# Support Vector Machine Prediction\n")
            f.write("Confusion matrix:\n")
            pos_pos = 0
            pos_gray = 0
            pos_neg = 0
            neg_pos = 0
            neg_gray = 0
            neg_neg = 0
            for prediction in prediction_list:
                (log_p, svc_p, comb_p, sam_p) = prediction.get_predictions()
                if prediction.sample.ihc:
                    if svc_p == "MSI":
                        pos_pos += 1
                    elif svc_p == "MSS":
                        pos_neg += 1
                    else:
                        pos_gray += 1
                else:
                    if svc_p == "MSI":
                        neg_pos += 1
                    elif svc_p == "MSS":
                        neg_neg += 1
                    else:
                        neg_gray += 1
            f.write("\tMSI\tGray\tMSS\n")
            f.write("dMMR\t{}\t{}\t{}\n".format(pos_pos, pos_gray, pos_neg))
            f.write("pMMR\t{}\t{}\t{}\n".format(neg_pos, neg_gray, neg_neg))
            f.write("\n")
            f.write("Accuracy: {}\n".format(sum([pos_pos, neg_neg]) / sum([pos_pos, pos_gray, pos_neg, neg_pos, neg_gray, neg_neg])))
            bacc = (pos_pos / sum([pos_pos, pos_gray, pos_neg]) + neg_neg / sum([neg_neg, neg_gray, neg_pos])) / 2
            f.write("Balanced Accuracy: {}\n".format(bacc))
            f.write("Precision MSI: {}\n".format(pos_pos / sum([pos_pos, neg_pos])))
            f.write("Recall MSI: {}\n".format(pos_pos / sum([pos_pos, pos_gray, pos_neg])))
            f.write("Precision MSS: {}\n".format(neg_neg / sum([neg_neg, pos_neg])))
            f.write("Recall MSS: {}\n".format(neg_neg / sum([neg_neg, neg_gray, neg_pos])))
            f.write("\n")
            
            f.write("# Voting/Combination Prediction\n")
            f.write("Confusion matrix:\n")
            pos_pos = 0
            pos_gray = 0
            pos_neg = 0
            neg_pos = 0
            neg_gray = 0
            neg_neg = 0
            for prediction in prediction_list:
                (log_p, svc_p, comb_p, sam_p) = prediction.get_predictions()
                if prediction.sample.ihc:
                    if comb_p == "MSI":
                        pos_pos += 1
                    elif comb_p == "MSS":
                        pos_neg += 1
                    else:
                        pos_gray += 1
                else:
                    if comb_p == "MSI":
                        neg_pos += 1
                    elif comb_p == "MSS":
                        neg_neg += 1
                    else:
                        neg_gray += 1
            f.write("\tMSI\tGray\tMSS\n")
            f.write("dMMR\t{}\t{}\t{}\n".format(pos_pos, pos_gray, pos_neg))
            f.write("pMMR\t{}\t{}\t{}\n".format(neg_pos, neg_gray, neg_neg))
            f.write("\n")
            f.write("Accuracy: {}\n".format(sum([pos_pos, neg_neg]) / sum([pos_pos, pos_gray, pos_neg, neg_pos, neg_gray, neg_neg])))
            bacc = (pos_pos / sum([pos_pos, pos_gray, pos_neg]) + neg_neg / sum([neg_neg, neg_gray, neg_pos])) / 2
            f.write("Balanced Accuracy: {}\n".format(bacc))
            f.write("Precision MSI: {}\n".format(pos_pos / sum([pos_pos, neg_pos])))
            f.write("Recall MSI: {}\n".format(pos_pos / sum([pos_pos, pos_gray, pos_neg])))
            f.write("Precision MSS: {}\n".format(neg_neg / sum([neg_neg, pos_neg])))
            f.write("Recall MSS: {}\n".format(neg_neg / sum([neg_neg, neg_gray, neg_pos])))
        
        # plot ROC curve
        log_list = list()
        svc_list = list()
        comb_list = list()
        true_list = list()
        for pred in prediction_list:
            if pred.sample.ihc is True or pred.sample.ihc is False:
                (log, svc, comb, _) = pred.get_prediction_values()
                (log_p, svc_p, comb_p, _) = pred.get_predictions()
                # (log, log_p, svc, svc_p, comb, comb_p, _, _) = self._predictor.sample_prediction(sample)
                if str(log) != 'nan' and str(svc) != 'nan' and str(comb) != 'nan':
                    true_list.append(pred.sample.ihc)
                    log_list.append(log)
                    svc_list.append(svc)
                    comb_list.append(comb)
                
        ns_prob = [0 for _ in range(len(true_list))]
        ns_fpr, ns_tpr, _ = roc_curve(true_list, ns_prob)
        logres_fpr, logres_tpr, _ = roc_curve(true_list, log_list)
        svc_fpr, svc_tpr, _ = roc_curve(true_list, svc_list)
        comb_fpr, comb_tpr, _ = roc_curve(true_list, comb_list)

        plt.figure(figsize=(20,20))
        plt.plot(ns_fpr, ns_tpr, label="No Prediction ({:.2f})".format(roc_auc_score(true_list, ns_prob)))
        plt.plot(logres_fpr, logres_tpr, label="LogisticRegression ({:.2f})".format(roc_auc_score(true_list, log_list)))
        plt.plot(svc_fpr, svc_tpr, label="SVC ({:.2f})".format(roc_auc_score(true_list, svc_list)))
        plt.plot(comb_fpr, comb_tpr, label="RegionCombination ({:.2f})".format(roc_auc_score(true_list, comb_list)))
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.savefig("{}/roc.png".format(output_dir))
        
