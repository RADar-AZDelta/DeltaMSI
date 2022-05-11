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

class Profile:

    def __init__(self, region: Region):
        """Create a new profile

        Args:
            region (Region): The region of this profile
        """
        self._region = region 
        self._length_dict = dict()
        self._depth = 0

    @property 
    def region(self) -> Region:
        """The Region of this Profile

        Returns:
            Region: The region
        """
        return self._region

    @property 
    def length_dict(self) -> dict:
        """A dict with as key the lengths, as values the depth

        Returns:
            dict: a dict
        """
        return self._length_dict

    @length_dict.setter 
    def length_dict(self, l_dict: dict):
        """Set the length dict

        Args:
            l_dict (dict): The new length dict
        """
        if type(l_dict) is dict:
            self._length_dict = l_dict

    @property
    def depth(self) -> int:
        """The total depth in this region

        Returns:
            int: The total depth
        """
        d = 0
        for l in self._length_dict:
            d = d + self._length_dict[l]
        return d

    def add_count(self, length: int):
        """Add a count at the given length

        Args:
            length (int): The given length
        """
        if length not in self._length_dict:
            self._length_dict[length] = 0
        self._length_dict[length] = self._length_dict[length] + 1
        self._depth = self._depth + 1

    def get_normalized_dict(self) -> dict:
        """Get a normalized dict on max coverage

        Returns:
            dict: The normalized dict
        """
        norm_dict = dict()
        max_depth = 1
        for length in self._length_dict:
            if self._length_dict[length] > max_depth:
                max_depth = self._length_dict[length]
        for length in self._length_dict:
            norm_dict[length] = self._length_dict[length] / max_depth
        return norm_dict

    def get_max_length(self) -> int:
        """Returns the found maximum length

        Returns:
            int: The maximum length
        """
        return max(self._length_dict.keys())

    def get_normalized_value(self, length: int) -> float:
        """Get the normalized depth (scaled) at a certain length position

        Args:
            length (int): The microsatelitte length to get the scaled depth for

        Returns:
            float: The scaled depth
        """
        try:
            return self.get_normalized_dict[length]
        except:
            return 0
    