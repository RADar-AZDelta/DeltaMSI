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

from deltamsi.model.region import Region
from deltamsi.model.profile import Profile

import json

def write_json(json_file: str, sample_name: str, flanking: int, minimum_mapping_quality: int, profile_dict: dict):
    """Create a json file of all the data

    Args:
        json_file (str): The path to the json file
        sample_name (str): The name of the sample
        flanking (int): The number of bp flanking the regions
        minimum_mapping_quality (int): The minimum mapping quality
        profile_dict (dict): The dict of regions and profiles
    """
    # create the json dict
    json_dict = dict()
    json_dict["sample_name"] = sample_name
    json_dict["flanking"] = flanking
    json_dict["minimum_mapping_quality"] = minimum_mapping_quality
    json_dict["regions"] = list()
    for region in profile_dict:
        profile = profile_dict[region]
        region_dict = dict()
        region_dict["name"] = region.name
        region_dict["region_name"] = region.to_string()
        region_dict["chr"] = region.chr
        region_dict["start"] = region.start 
        region_dict["end"] = region.end
        region_dict["length"] = region.length
        region_dict["depth"] = profile.depth
        region_dict["profile"] = profile.length_dict
        region_dict["norm_profile"] = profile.get_normalized_dict()
        json_dict["regions"].append(region_dict)
    # write to a file
    with open(json_file, 'w') as outfile:
        json.dump(json_dict, outfile)

def dict_to_json(json_file: str, json_dict: dict):
    """Create a json file from a dict

    Args:
        json_file (str): The path to the file
        json_dict (dict): The dict to use for the json file
    """
    # write to a file
    with open(json_file, 'w') as outfile:
        json.dump(json_dict, outfile)

        
def read_json(json_file: str) -> dict:
    """Read a json file of the data

    Args:
        json_file (str): The path to the json file

    Returns:
        dict: The json file as a dict
    """
    with open(json_file, 'r') as infile:
        json_dict = json.load(infile)
    return json_dict