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

def create_regions_from_bed(bed_file: str) -> list:
    """Create a list of regions from the given bed file

    Args:
        bed_file (str): The bed file to parse

    Returns:
        list: A list of regions
    """
    regions = list()
    if (bed_file is not None):
        #create regions from bed_file
        try:
            f = open(bed_file, 'r')
            for line in f:
                line = line.strip()
                columns=line.split("\t")
                chr = columns[0]
                start = int(columns[1])
                end = int(columns[2])
                name = None
                if (len(columns) > 3):
                    name = columns[3]
                regions.append(Region(chr, start, end, name))
        except IOError:
            print("The bed file gives an error")
            raise
        finally:
            f.close()
    return regions