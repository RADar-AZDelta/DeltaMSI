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

from model.sample import Sample
import numpy as np

class Prediction():

    def __init__(self, sample: Sample):
        """Creates a new Prediction object

        Args:
            sample (Sample): The sample for this prediction
        """
        self._sample = sample
        self._region_dict = dict()
        self._final_value_dict = dict()
        self._final_prediction_dict = dict()
        for col in ["LogisticRegression", "SVC", "Voting", "Sample"]:
            self._final_value_dict[col] = np.nan
            self._final_prediction_dict[col] = None

    @property 
    def sample(self) -> Sample:
        """Get the sample of this prediction

        Returns:
            Sample: The sample this prediction belongs to
        """
        return self._sample

    def set_prediction(self, logres: float, logres_pred: str, 
                svc: float, svc_pred: str, voting: float, voting_pred: str, 
                sample: float, sample_pred: str):
        """Set all prediciton values

        Args:
            logres (float): The raw logres value
            logres_pred (str): The interpretation of the logres value
            svc (float): The raw svm value
            svc_pred (str): The interpretation of the svm value
            voting (float): The raw voting value
            voting_pred (str): The interpretation of the voting value
            sample (float): The raw sample value
            sample_pred (str): The interpretation of the sample value
        """
        self._final_value_dict["LogisticRegression"] = logres
        self._final_prediction_dict["LogisticRegression"] = logres_pred
        self._final_value_dict["SVC"] = svc
        self._final_prediction_dict["SVC"] = svc_pred
        self._final_value_dict["Voting"] = voting
        self._final_prediction_dict["Voting"] = voting_pred
        self._final_value_dict["Sample"] = sample
        self._final_prediction_dict["Sample"] = sample_pred

    def get_prediction_values(self):
        """Returns the raw values

        Returns:
            (float, float, float, float): LogisticRegression, SupportVectorMachine, Voting, Sample
        """
        return (self._final_value_dict["LogisticRegression"], 
                    self._final_value_dict["SVC"],
                    self._final_value_dict["Voting"],
                    self._final_value_dict["Sample"])

    def get_predictions(self):
        """Returns the interpretations of the predictions

        Returns:
            (str, str, str, str): LogisticRegression, SupportVectorMachine, Voting, Sample
        """
        return (self._final_prediction_dict["LogisticRegression"], 
                    self._final_prediction_dict["SVC"],
                    self._final_prediction_dict["Voting"],
                    self._final_prediction_dict["Sample"])

    def add_region(self, region_name: str, logres: float, svc: float, voting: float):
        """Add a region to the prediction

        Args:
            region_name (str): The name of the region
            logres (float): The outcome of the Logistic Regression model
            svc (float): The outcome of the Support Vector Machine
            voting (float): The outcome of the Voting model
        """
        self._region_dict[region_name] = dict()
        self._region_dict[region_name]["LogisticRegression"] = logres
        self._region_dict[region_name]["SVC"] = svc
        self._region_dict[region_name]["Voting"] = voting

    def get_region(self, region_name: str):
        """Get the results of a given region

        Args:
            region_name (str): The name of the region

        Returns:
            (float, float, float): LogisticRegression, SupportVectorMachine, VotingModel
        """
        if region_name in self._region_dict:
            return (self._region_dict[region_name]["LogisticRegression"],
                        self._region_dict[region_name]["SVC"], 
                        self._region_dict[region_name]["Voting"])
        else:
            return (np.nan, np.nan, np.nan)