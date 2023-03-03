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

def update_ihc_status(ihc_file: str, sample_dict: dict) -> dict:
    """Read the IHC status and update the samples

    Args:
        ihc_file (str): The path to the ihc file
        sample_dict (dict): The directory of samples (key=sample_name)

    Returns:
        dict: The updated sample_dict
    """
    if (ihc_file is not None):
        #create regions from bed_file
        try:
            f = open(ihc_file, 'r')
            for line in f:
                line = line.strip()
                splitter = ","
                if "\t" in line:
                    splitter = "\t"
                columns=line.split(splitter)
                #(pMMR/dMMR, 0/1 or MSS/MSI)
                if (len(columns) >= 2):
                    name = columns[0]
                    value = columns[1]
                    ihc = None 
                    if str(value).lower() == "pmmr" or str(value).lower() == "0" or str(value).lower() == "mss":
                        ihc = False
                    if str(value).lower() == "dmmr" or str(value).lower() == "1" or str(value).lower() == "msi":
                        ihc = True
                    if name in sample_dict:
                        sample_dict[name].ihc = ihc
        except IOError:
            print("The ihc file gives an error")
            raise
        finally:
            f.close()
    return sample_dict