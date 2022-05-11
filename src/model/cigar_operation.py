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

from enum import Enum

class Cigar_Operation(Enum):
    """An enum of the cigar operations defined in pysam
    """
    BAM_CMATCH = (0, "M")
    BAM_CINS = (1, "I")
    BAM_CDEL = (2, "D")
    BAM_CREF_SKIP = (3, "N")
    BAM_CSOFT_CLIP = (4, "S")
    BAM_CHARD_CLIP = (5, "H")
    BAM_CPAD = (6, "P")
    BAM_CEQUAL = (7, "=")
    BAM_CDIFF = (8, "X")
    BAM_CBACK = (9, "B")

    def __init__(self, number: int, name: str):
        """Create a cigar operation

        Args:
            number (int): The number of the operation
            name (str): The name of the operation
        """
        self.number = int(number)
        self.letter = str(name)

    @staticmethod
    def get_cigar_operation_with_number(number: int) -> 'Cigar_Operation':
        """Get the operation based on the number

        Args:
            number (int): The number of the operation

        Returns:
            Cigar_Operation: The operation
        """
        for name, member in Cigar_Operation.__members__.items():
            if (int(number) == int(member.number)):
                return member
        return None

    @staticmethod
    def get_cigar_operation_with_letter(letter: str) -> 'Cigar_Operation':
        """Get the operation based on the letter

        Args:
            letter (str): The letter of the operation

        Returns:
            Cigar_Operation: The operation
        """
        for name, member in Cigar_Operation.__members__.items():
            if (str(letter) == str(member.letter)):
                return member
        return None