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
import argparse
import sys 
import os
from deltamsi.ai.create_profile import create_profile
from deltamsi.ai.model import AIModel

def main():
    parser = argparse.ArgumentParser(description="""
    DeltaMSI Copyright 2022 Koen Swaerts, AZ Delta
    This program comes with ABSOLUTELY NO WARRANTY; for details see the GPLv3 LICENCE
    This is free software, and you are welcome to redistribute it
    under certain conditions;
    """)

    subparsers = parser.add_subparsers(dest='type', help='The core command to execute')
    # parser_profile = subparsers.add_parser('profile', help = "Profile a sample (only creates json files of a sample, used in development)")
    # parser_profile.add_argument("bed_file", help='The bed file of the regions (chr,start,end,name)')
    # parser_profile.add_argument("bam_file", help='The bam file of the sample')
    # parser_profile.add_argument("--flanking", "-f", help='The number of bases the flanking must use', default=5, type=float)
    # parser_profile.add_argument("--minimum_mapping_quality", "-m", help='The minimum mapping quality of the reads', default=20, type=float)
    # parser_profile.add_argument("--json_file", "-json", help='The output json file')
    # parser_profile.add_argument('-v', "--verbose", help='verbose', action='store_true')
    
    parser_profile = subparsers.add_parser('train', help = "Train a new model")
    parser_profile.add_argument("--bed_file", "-bed", help='The bed file of the regions (chr,start,end,name)')
    parser_profile.add_argument("--ihc_file", "-ihc", help='Text file (tsv or csv) with as first column the sample_name, second ihc value (pMMR/dMMR, 0/1 or MSS/MSI)')
    parser_profile.add_argument("--bam_file", "-bam", help='The bam files of the samples', action='append')
    parser_profile.add_argument("--bam_list_file", "-bamf", help='A file with all complete paths to the bam files of the samples')
    parser_profile.add_argument("--flanking", "-f", help='The number of bases the flanking must use', default=5, type=int)
    parser_profile.add_argument("--minimum_mapping_quality", "-m", help='The minimum mapping quality of the reads', default=20, type=int)
    parser_profile.add_argument("--depth", "-d", help='The minimum dapth of a region', default=30, type=int)
    parser_profile.add_argument("--out_dir", "-o", help='The output directory for the model')
    parser_profile.add_argument('-v', "--verbose", help='verbose', action='store_true')
    
    parser_profile = subparsers.add_parser('predict', help = "Predict one or multiple samples")
    parser_profile.add_argument("--model_directory", "--model", "-m", help='The model to use')
    parser_profile.add_argument("--bam_file", "-bam", help='The bam files of the samples', action='append')
    parser_profile.add_argument("--bam_list_file", "-bamf", help='A file with all complete paths to the bam files of the samples')
    parser_profile.add_argument("--out_dir", "-o", help='The output directory for the results')
    parser_profile.add_argument('-v', "--verbose", help='verbose', action='store_true')
    
    parser_profile = subparsers.add_parser('evaluate', help = "Evaluate the model with known data")
    parser_profile.add_argument("--model_directory", "--model", "-m", help='The model to use')
    parser_profile.add_argument("--bam_file", "-bam", help='The bam files of the samples', action='append')
    parser_profile.add_argument("--bam_list_file", "-bamf", help='A file with all complete paths to the bam files of the samples')
    parser_profile.add_argument("--out_dir", "-o", help='The output directory for the results')
    parser_profile.add_argument("--ihc_file", "-ihc", help='Text file (tsv or csv) with as first column the sample_name, second ihc value (pMMR/dMMR, 0/1 or MSS/MSI)')
    parser_profile.add_argument('-v', "--verbose", help='verbose', action='store_true')
    

    args = parser.parse_args()
    type = args.type

    if type == "profile":
        bam_file = args.bam_file
        bed_file = args.bed_file
        flanking = args.flanking
        minimum_mapping_quality = args.minimum_mapping_quality
        json_file = args.json_file
        verbose = args.verbose
        if json_file is None:
            print("No output file selected")
            sys.exit("No output file selected")
        create_profile(bam_file, bed_file, flanking, minimum_mapping_quality, json_file)
    elif type == "train":
        bam_list = args.bam_file
        bam_list_file = args.bam_list_file
        bed_file = args.bed_file
        ihc_file = args.ihc_file
        flanking = args.flanking
        depth = args.depth
        minimum_mapping_quality = args.minimum_mapping_quality
        out_dir = args.out_dir
        verbose = args.verbose
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        if out_dir is None:
            print("No output directory selected")
            sys.exit("No output directory selected")
        if bed_file is None or (not os.path.exists(bed_file)):
            print("No bed file selected")
            sys.exit("No bed file selected")
        if ihc_file is None or (not os.path.exists(ihc_file)):
            print("No ihc file selected")
            sys.exit("No ihc file selected")
        if bam_list_file is not None and os.path.exists(bam_list_file):
            if bam_list is None:
                bam_list = list()
            try:
                f = open(bam_list_file, 'r')
                for line in f:
                    line = line.strip()
                    bam_list.append(line)
            except IOError:
                print("The bam list file gives an error")
                raise
            finally:
                f.close()
        correct_bam_list = list()
        for bam_file in bam_list:
            if bam_file is None or (not os.path.exists(bam_file)):
                print("Not found bam file: {}".format(bam_file))
            else:
                correct_bam_list.append(bam_file)
        if len(correct_bam_list) == 0:
            print("No bam file selected")
            sys.exit("No bam file selected")
        AIModel.create_model(bed_file, correct_bam_list, ihc_file, out_dir, 
                    flanking, minimum_mapping_quality, depth, verbose=verbose)
    elif type == "predict":
        bam_list = args.bam_file
        bam_list_file = args.bam_list_file
        model_directory = args.model_directory
        out_dir = args.out_dir
        verbose = args.verbose
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        if out_dir is None:
            print("No output directory selected")
            sys.exit("No output directory selected")
        if model_directory is None or (not os.path.exists(model_directory)):
            print("No model file selected")
            sys.exit("No model file selected")
        if bam_list_file is not None and os.path.exists(bam_list_file):
            if bam_list is None:
                bam_list = list()
            try:
                f = open(bam_list_file, 'r')
                for line in f:
                    line = line.strip()
                    bam_list.append(line)
            except IOError:
                print("The bam list file gives an error")
                raise
            finally:
                f.close()
        correct_bam_list = list()
        for bam_file in bam_list:
            if bam_file is None or (not os.path.exists(bam_file)):
                print("Not found bam file: {}".format(bam_file))
            else:
                correct_bam_list.append(bam_file)
        if len(correct_bam_list) == 0:
            print("No bam file selected")
            sys.exit("No bam file selected")
        AIModel.predict(model_directory, correct_bam_list, out_dir, verbose=verbose)
    elif type == "evaluate":
        bam_list = args.bam_file
        bam_list_file = args.bam_list_file
        model_directory = args.model_directory
        ihc_file = args.ihc_file
        out_dir = args.out_dir
        verbose = args.verbose
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        if out_dir is None:
            print("No output directory selected")
            sys.exit("No output directory selected")
        if model_directory is None or (not os.path.exists(model_directory)):
            print("No model file selected")
            sys.exit("No model file selected")
        if ihc_file is None or (not os.path.exists(ihc_file)):
            print("No ihc file selected")
            sys.exit("No ihc file selected")
        if bam_list_file is not None and os.path.exists(bam_list_file):
            if bam_list is None:
                bam_list = list()
            try:
                f = open(bam_list_file, 'r')
                for line in f:
                    line = line.strip()
                    bam_list.append(line)
            except IOError:
                print("The bam list file gives an error")
                raise
            finally:
                f.close()
        correct_bam_list = list()
        for bam_file in bam_list:
            if bam_file is None or (not os.path.exists(bam_file)):
                print("Not found bam file: {}".format(bam_file))
            else:
                correct_bam_list.append(bam_file)
        if len(correct_bam_list) == 0:
            print("No bam file selected")
            sys.exit("No bam file selected")
        AIModel.evaluate(model_directory, correct_bam_list, ihc_file, out_dir, verbose=verbose)


if __name__ == "__main__":
    main()