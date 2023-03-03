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
from deltamsi.model.region_profile import RegionProfile

class Sample():
    
    def __init__(self, name: str):
        """Create a new Sample

        Args:
            name (str): the name of the sample
        """
        self.name = name
        self.region_dict = dict()
        self.ihc= None
        
    def add_region(self, region: RegionProfile):
        """Add a new region to this sample

        Args:
            region (RegionProfile): The region to add
        """
        self.region_dict[region.name] = region
        
    def get_region(self, region_name: str) -> RegionProfile:
        """Get a region, based on name

        Args:
            region_name (str): The name of the region

        Returns:
            RegionProfile: The region
        """
        return self.region_dict.get(region_name)
    
    def get_number_of_regions_above_depth(self, depth: int) -> int:
        """Get the number of useable regions above the given depth

        Args:
            depth (int): The depth

        Returns:
            int: The number of regions
        """
        nr = 0
        for region_name in self.region_dict:
            if self.region_dict[region_name].depth >= depth:
                nr+=1
        return nr
    
    def get_region_above_depth(self, region_name: str, depth: int) -> RegionProfile:
        """Get a region, if it is above the depth (None otherwise)

        Args:
            region_name (str): The name of the region
            depth (int): The needed depth

        Returns:
            RegionProfile: The region (or None)
        """
        region = None
        if region_name in self.region_dict:
            if self.region_dict[region_name].depth >= depth:
                region = self.region_dict[region_name]
        return region

    def __eq__(self, o):
        if self.name == o.name:
            return True
        else:
            return False