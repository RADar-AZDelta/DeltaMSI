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

class Region():

    def __init__(self, chr: str, start: int, end: int, name: str=None):
        """Create a new Region

        Args:
            chr (str): The chromosome
            start (int): The start of the region
            end (int): The end of the region
            name (str, optional): The name of th region. Defaults to None.
        """
        self._chr = chr
        self._start = start
        self._end = end
        self._name = name

    @property
    def chr(self) -> str:
        """Get the chromosome

        Returns:
            str: The chromosome
        """
        return self._chr 

    @property 
    def start(self) -> int:
        """Get the start of the region

        Returns:
            int: The start
        """
        return self._start 

    @property 
    def end(self) -> int:
        """Get the end of the region

        Returns:
            int: The end
        """
        return self._end

    @property 
    def name(self) -> str:
        """Get the regions name

        Returns:
            str: The name
        """
        return self._name

    @property
    def length(self) -> int:
        """Get the length of this region

        Returns:
            int: the length
        """
        return self.end - self.start

    def to_string(self) -> str:
        """Turns the region to a string

        Returns:
            str: The region
        """
        return "{}:{}-{}".format(self.chr, self.start, self.end)
        
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