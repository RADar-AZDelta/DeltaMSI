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

import numpy as np

class RegionProfile():
    
    def __init__(self, name: str, chr: str, start: int, end: int, 
                 depth: int, norm_profile_dict: dict, profile_dict: dict, 
                 flanking: int=0):
        """Create a new Profile of the region

        Args:
            name (str): The name of the region
            chr (str): The chromosome
            start (int): The start
            end (int): The end
            depth (int): The sequencing depth
            norm_profile_dict (dict): A normalised dict of the profile
            profile_dict (dict): The actual profile
            flanking (int, optional): The number of bases flanking the region. Defaults to 0.
        """
        self.name = name
        self.chr = chr
        self.start = start
        self.end = end
        self.depth = depth
        self.flanking = int(flanking)
        self.norm_profile_dict = norm_profile_dict
        self.profile_dict = profile_dict
        
    @property
    def length(self) -> int:
        """Returns the actual length of the region

        Returns:
            int: the length
        """
        return self.end - self.start
    
    @property
    def region_length(self) -> int:
        """Returns the length that must be used for the model (adding extra flanking bases)

        Returns:
            int: The length used in the model
        """ 
        return (self.end - self.start) + (4* self.flanking)
    
    def get_graph(self, cutoff: float=0, flex_cutoff: bool=True) -> list:
        """Generates a list as graph, each element contains the number of reads with that length

        Args:
            cutoff (float, optional): The cutoff to use. Defaults to 0.
            flex_cutoff (bool, optional): Use the flex cutoff. Defaults to True.

        Returns:
            list: The graph that must be shown
        """
        graph_list = list()
        for i in range(0, self.region_length+1):
            graph_list.append(self.get_value_norm_on(i, 
                                    cutoff=cutoff, flex_cutoff=flex_cutoff))
        return graph_list
    
    def get_flex_cutoff(self) -> float:
        """Calcultate the flexible cutoff

        Returns:
            float: The flexible cutoff
        """
        # Calculate the flexible cutoff
        if self.depth == 0:
            return 0.0001
        return (1/self.get_peak_depth()) + 0.01
    
    def get_recalced_depth(self, cutoff: float=0, flex_cutoff: bool=True) -> int:
        """Recalculate the depth based on the given cutoff

        Args:
            cutoff (float, optional): The hard cutoff. Defaults to 0.
            flex_cutoff (bool, optional): Use the flexible cutoff. Defaults to True.

        Returns:
            int: The depth with noise removal
        """
        # recalculate the depth of the graph, based on the cutoff
        if flex_cutoff:
            cutoff = self.get_flex_cutoff()
        d = 0
        for l in self.norm_profile_dict:
            val = self.get_value_norm_on(l)
            if val > cutoff:
                d += self.get_value_actual_on(l)
        return d
    
    def get_msings(self, cutoff: float=0, flex_cutoff: bool=True) -> int:
        """Calculates the msings score

        Args:
            cutoff (float, optional): The cutoff to use. Defaults to 0.
            flex_cutoff (bool, optional): Use flexible cutoff. Defaults to True.

        Returns:
            int: The number of msings peaks
        """
        # calculate the msings score
        if flex_cutoff:
            cutoff = self.get_flex_cutoff()
        msings = 0
        for l in self.norm_profile_dict:
            val = self.get_value_norm_on(l)
            if val >= cutoff:
                msings+=1
        return msings
    
    def get_peak_depth(self) -> int:
        """Get the number of reads on the peak of the graph

        Returns:
            int: The number of reads
        """
        # get the depth of the highest peak
        for l in self.norm_profile_dict:
            val = self.get_value_norm_on(l)
            if val == 1:
                return self.get_value_actual_on(l)
        return np.nan
    
    def get_value_norm_on(self, length: int, cutoff: float=0, flex_cutoff: bool=False) -> float:
        """Get the normalised value on a given length

        Args:
            length (int): The length
            cutoff (float, optional): The hard cutoff to use. Defaults to 0.
            flex_cutoff (bool, optional): Use the flexible cutoff. Defaults to True.

        Returns:
            float: The normalised value on that length, or 0 if below the cutoff
        """
        if flex_cutoff:
            cutoff = self.get_flex_cutoff()
        # get the normalised value on this position
        if int(length) < int(self.flanking):
            return 0
        if str(length) in self.norm_profile_dict:
            if float(self.norm_profile_dict[str(length)]) < cutoff:
                return 0
            else:
                return float(self.norm_profile_dict[str(length)])
        elif int(length) in self.norm_profile_dict:
            if float(self.norm_profile_dict[int(length)]) < cutoff:
                return 0
            else:
                return float(self.norm_profile_dict[int(length)])
        else:
            return 0
    
    def get_value_actual_on(self, length: int) -> int:
        """Get the real number of reads with the given length

        Args:
            length (int): The length

        Returns:
            int: The number of reads
        """
        # get the actual depth on this position
        if int(length) < int(self.flanking):
            return 0
        if str(length) in self.profile_dict:
            return float(self.profile_dict[str(length)])
        elif int(length) in self.profile_dict:
            return float(self.profile_dict[int(length)])
        else:
            return 0
        
    def get_min_max(self):
        """Get the minimum and maximum length of the observed microsatelitte

        Returns:
            (int, int): The minimum and maximum length
        """
        # get the minimum and maximum lengths in the dicts
        minimum = 1000
        maximum = -10000
        for l in self.norm_profile_dict:
            if int(l) > maximum:
                maximum = int(l)
            if int(l) < minimum:
                minimum = int(l)
        return (minimum, maximum)

    @staticmethod
    def from_profile(profile: Profile, flanking: int) -> 'RegionProfile':
        """Create a RegionProfile from a Profile

        Args:
            profile (Profile): The profile to use
            flanking (int): The flanking to use

        Returns:
            RegionProfile: The RegionProfile object
        """
        region_profile = RegionProfile(profile.region.name, profile.region.chr, 
            profile.region.start, profile.region.end, profile.depth,
            profile.get_normalized_dict(), profile.length_dict, flanking=flanking)
        return region_profile

    @staticmethod
    def from_dict(raw_dict: dict, flanking: int) -> 'RegionProfile':
        """Create a RegionProfile from a dict (like from a save json profile)

        Args:
            raw_dict (dict): The dict
            flanking (int): The flanking to use

        Returns:
            RegionProfile: The RegionProfile object
        """
        region_name = raw_dict["name"]
        region_chr = raw_dict["chr"]
        region_start = raw_dict["start"]
        region_end = raw_dict["end"]
        region_depth = raw_dict["depth"]
        region_norm_profile = raw_dict["norm_profile"]
        region_profile = raw_dict["profile"]
        return RegionProfile(region_name, region_chr, region_start, region_end, 
                        region_depth, region_norm_profile, region_profile,
                        flanking=flanking)

        
    def __lt__(self, other):
        if self.chr == other.chr:
            if int(self.start) == int(other.start):
                return int(self.end) < int(other.end)
            else:
                return int(self.start) < int(other.start)
        else:
            schr = self.chr.replace("chr", "")
            ochr = other.chr.replace("chr", "")
            if schr == "X":
                return False
            if ochr == "X":
                return True
            return int(schr) < int(ochr)


    def __eq__(self, other):
        if self.chr == other.chr and int(self.start) == int(other.start) and int(self.end) == int(other.end):
            return True
        else:
            return False

    def __hash__(self):
        return hash("{}:{}-{}".format(self.chr, self.start, self.end))