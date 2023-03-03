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
from deltamsi.io_module.bam_files import parse_bam_file_to_profiles
from deltamsi.io_module.json_files import write_json
from deltamsi.io_module.json_files import read_json

from deltamsi.model.region import Region 
from deltamsi.model.profile import Profile
from deltamsi.model.sample import Sample
from deltamsi.model.region_profile import RegionProfile

import os

def create_profile(bam_file: str, bed_file: str, flanking: int, minimum_mapping_quality: int, json_file: str):
    """Create a profile json

    Args:
        bam_file (str): The path to the bam file
        bed_file (str): The path to the bed file
        flanking (int): The number of flanking bases
        minimum_mapping_quality (int): The minimum mapping quality
        json_file (str): The path to the output json file
    """
    # get the regions
    regions = create_regions_from_bed(bed_file)
    # parse the bam file to profiles
    profile_dict = parse_bam_file_to_profiles(bam_file, regions, flanking, minimum_mapping_quality)
    # save the profiles in a json file
    file_name = os.path.basename(bam_file)
    sample_name = os.path.splitext(file_name)[0]
    write_json(json_file, sample_name, flanking, minimum_mapping_quality, profile_dict)

def get_sample_from_bam(bam_file: str, regions: set, flanking: int, minimum_mapping_quality: int) -> Sample:
    """Create a sample object from a bam file

    Args:
        bam_file (str): The path to the bam file
        regions (set): A set of all regions to screen
        flanking (int): The number of bases to use for flanking the regions
        minimum_mapping_quality (int): The minimum mapping quality of reads to use

    Returns:
        Sample: A sample object with all region information
    """
    # parse the bam file to sample object
    file_name = os.path.basename(bam_file)
    sample_name = os.path.splitext(file_name)[0]
    profile_dict = parse_bam_file_to_profiles(bam_file, regions, flanking, minimum_mapping_quality)
    sample = Sample(sample_name)
    for region_name in profile_dict:
        profile = profile_dict[region_name]
        regionprofile = RegionProfile.from_profile(profile, flanking)
        sample.add_region(regionprofile)
    return sample


