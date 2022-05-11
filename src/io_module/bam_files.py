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

from model.region import Region
from model.profile import Profile
from model.cigar_operation import Cigar_Operation

import pysam


def get_length_from_read(read: 'pysam.AlignmentSegment', region: Region, flanking: int) -> int:
    """Create a length from the read, the read must be filtered before the use of this method!

    Args:
        read (pysam.AlignmentSegment): The read
        region (Region): The region object to check
        flanking (int): The length of flanking before and after the region

    Returns:
        int: The length of this read
    """
    longcigar = ""
    for (code, count) in read.cigartuples:
        cigar_op = Cigar_Operation.get_cigar_operation_with_number(code)
        for i in range(0, count):
            longcigar = "{}{}".format(longcigar, cigar_op.letter)
    read_index = 0
    cigar_index = 0
    ref_index = read.reference_start
    region_length = 0
    region_cigar = ""
    for i in range(0, read.query_length):
        cigar_op = Cigar_Operation.get_cigar_operation_with_letter(longcigar[i])
        in_region = (region.start - flanking <= ref_index) and (ref_index < region.end + flanking)
        if in_region:
            region_cigar = "{}{}".format(region_cigar, longcigar[i])
        if cigar_op == Cigar_Operation.BAM_CHARD_CLIP or cigar_op == Cigar_Operation.BAM_CPAD:
            #hard clipped or padded base
            cigar_index += 1
        if cigar_op == Cigar_Operation.BAM_CSOFT_CLIP:
            #soft clipped base
            cigar_index += 1
            read_index += 1
        if cigar_op == Cigar_Operation.BAM_CMATCH or cigar_op == Cigar_Operation.BAM_CEQUAL or cigar_op == Cigar_Operation.BAM_CDIFF:
            #base both on query and ref (match, same or diff)
            cigar_index += 1
            read_index += 1
            ref_index += 1
            if in_region:
                region_length += 1
        if cigar_op == Cigar_Operation.BAM_CREF_SKIP:
            #position skipped in ref
            ref_index += 1
        if cigar_op == Cigar_Operation.BAM_CINS:
            #INSERTION
            cigar_index += 1
            read_index += 1
            if in_region:
                region_length += 1
        if cigar_op == Cigar_Operation.BAM_CDEL:
            #DELETION
            cigar_index += 1
            ref_index += 1
    return region_length


def parse_to_profile(samfile: 'pysam.AlignmentFile', region: Region, flanking: int, minimum_mapping_quality: int) -> Profile:
    """Parse all reads that begin and end outside the region + flanking bases. The reads are filtered based on duplicates, supplementary and mapping quality.

    Args:
        samfile (pysam.AlignmentFile): The sam file object
        region (Region): The region to parse
        flanking (int): The number of bases to use as flanking the region
        minimum_mapping_quality (int): The minimum mapping quality to use

    Returns:
        Profile: the profile of the given region 
    """
    # profile = Profile(region, flanking=flanking)
    profile = Profile(region)
    for read in samfile.fetch(region.chr, region.start - flanking, region.end + flanking):
        #go over every read, that overlaps or starts/ends in the region
        if read.reference_start < (region.start - flanking) and read.reference_end > (region.end + flanking):
            #select only reads that are overlapping the region + flanking
            if not read.is_duplicate and not read.is_supplementary and read.mapping_quality >= minimum_mapping_quality:
                #removes duplicates, secondary reads and low mapping quality
                region_length = get_length_from_read(read, region, flanking)
                profile.add_count(region_length)
    return profile

def parse_bam_file_to_profiles(bam_file_path: str, regions: list, flanking: int, minimum_mapping_quality: int) -> dict:
    """Create a dict of profiles of all regions

    Args:
        bam_file_path (str): The path to the bam file
        regions (list): The list of regions
        flanking (int): The number of bases that are flanking the region
        minimum_mapping_quality (int): The minimum mapping quality to use

    Returns:
        dict: a dict with as key the region, as values the profile
    """
    profile_dict = dict()
    # read the sam file
    samfile = pysam.AlignmentFile(bam_file_path, "rb")
    for region in sorted(regions):
        # parse the regions into profiles
        profile = parse_to_profile(samfile, region, flanking, minimum_mapping_quality)
        profile_dict[region] = profile
    samfile.close()
    return profile_dict