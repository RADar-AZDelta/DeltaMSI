# DeltaMSI

DeltaMSI is a MSI screening tool using AI to score regions and samples.

## Citation

The article of this tool is currently under review.

## Licence

DeltaMSI  Copyright (C) 2022  Koen Swaerts, AZ Delta  
This program comes with ABSOLUTELY NO WARRANTY; for details see LICENSE  
This is free software, and you are welcome to redistribute it under certain conditions; for details see LICENSE  

## Install

DeltaMSI is written in Python3.9. It can be installed by using the Pipfile.  

### If you are using a Python virtual environment

To install DeltaMSI in a Python virtual environment do:

```bash
pip3 install --no-cache-dir --user pipenv
python -m pipenv install
```

### If you are using Containers like Docker or Podman

Podman and Docker can be used in the same way:

```bash
# Creation of the image
podman build -t deltamsi:v1.0.0 -f Dockerfile
# Check if the image is able to run
podman run --rm -it localhost/deltamsi:v1.0.0 -h
# this should show the general help
podman run --rm -it localhost/deltamsi:v1.0.0 train -h
# this should show the help for the train module
```

## Usage

DeltaMSI has 3 modes: training, prediction and evaluation.

### Training

```bash
python3 -m pipenv run python src/app.py train -bed /data/msi.bed -ihc /data/train_data.csv -o /data/deltamsi -bamf /data/train_samples.txt -v
```

In this step, based on the supplied regions and samples, regions and samples are filtered, models trained and cutoffs generated. Everything happens automatically, without user interaction.  
1. For each supplied region and sample the microsatellite length profile is determined. This is done by filtering all reads out of the bam files that overlap the complete mirosatellite, including a small flanking region at both sides to compensate possible missmapping. These reads are filtered for mapping quality and reads marked as duplicates are removed.  
1. The samples are split (stratified) in 2/3 training set, and 1/3 validation set.
1. Regions are filtered, based on the occurance in the training population. The regions must occure in minimum 75% of the samples, samples must have minimum 75% of the filtered regions covered. If samples were removed, both sets remain unchanged.  
1. Regions are scaled, for optimal training and prediction performance. Scaling is done by retrieving the length with the highest depth, and dividing all possible depth per length by this value. For each possible length of the microsatelitte a scaled depth is obtained (which is a value in the interval [0, 1])
1. Logistic Regression and Support Vector Machine models per region are build, with an automated hyperparameter tuning with a k-fold of 3. For all folds, the same sets (sample split) are used. Used features in the models are the scaled depth per possible length. This results in 2*#regions models.  
1. Based on the validation set regions are filtered according to the balanced accuracy (gives equal weights to the inbalanced MSS/MSI set). Only regions with a balanced accuracy above 0.6 are used.  
1. A new model is introduced: the Voting/combination model. This model uses the previous build Logistic Regression and Support Vector Machine to make a prediction. If both models predict a region as instable, the region will be predicted as instable. If one of the two models predict as stable, this model will predict stable.  
1. Sample scores for each model are calculated with the formula: *# instable regions / # passed filter regions*.   
1. The validation set is used to determine the 2 cut-offs used by DeltaMSI to predict the samples. Cut-offs are generated for the 3 models (Logistic Regression, Support Vector Machine and Voting/Combination model). For the first cut-off, this is done to get a 90% precission on the MSS prediction. For the second cut-off, we get a 90% precission for the MSI predicted samples. Between these cut-offs a gray zone is defined, suggesting to clinically check with another tool.  

#### Parameters

* `--bed_file`, `-bed`  The bed file of the regions (chr, start, end, name)  
* `--ihc_file`, `-ihc`  A text file (tsv or csv) with as first column the sample_name, second the IHC value (pMMR/dMMR, 0/1 or MSS/MSI)  
* `--bam_file`, `-bam`  The bam files of the samples, this can be used multiple times (and in combination with --bam_list_file)  
* `--bam_list_file`, `-bamf`    A file with all complete paths to the bam files (each new path is a new line)  
* `--flanking`, `-f`    The number of bases flanking the microsatellite region (default 5)  
* `--minimum_mapping_quality`, `-m` The minimum mapping quality to filter the reads (default 20)  
* `--depth`, `-d`   The minimum depth before the region is used (default 30)  
* `--out_dir`, `-o` The output directory for the model and results  
* `--verbose`, `-v` Write information about the process to the stdout  

**Note:** bam files must have as name: sample_name.bam. All bam files must be indexed (.bam.bai or .bai) to be able to be processed.  

#### Result files

* `aimodel.deltamsi`    The file containing all information about the models, filtering, ...   
* `cutoff_matrix_*.png`   Images to visualise the cutoffs and the used samples vs these cutoffs.  
* `cutoffs.txt`   A file containing the cut-off values for each model.  
* `regions.tsv`   A file with the information of all given regions (first 4 columns can be used as a bed file)  
* `regions.xslx`  Same as regions.tsv, but as Excel file  
* `regions.txt`   Description of the used regions, and removed regions  
* `roc.png`   The ROC curve of all models.  
All other files are the same as the result files for the prediction.

### Prediction

In this step one or multiple samples are predicted. 

```bash
# Predicting a single sample
python3 -m pipenv run python src/app.py predict -m /data/deltamsi/ -o /data/deltamsi_prediction --bam /data/mapped_data/sample1.bam -v
# Predicting multiple samples, giving multiple bam files
python3 -m pipenv run python src/app.py predict -m /data/deltamsi/ -o /data/deltamsi_prediction --bam /data/mapped_data/sample1.bam --bam /data/mapped_data/sample2.bam -v
# Predicting multiple samples, giving a file paths txt file
python3 -m pipenv run python src/app.py predict -m /data/deltamsi/ -o /data/deltamsi_prediction -bamf /data/bam_file_paths.txt -v
```

#### Parameters

* `--model_directory`, `--model`, `-m`  The path to the model directory (the --out_dir in the training step) or the path to the aimodel.deltamsi file (no other files are needed for the prediction).  
* `--bam_file`, `-bam`  The bam files of the sample(s) to predict, this can be used multiple times (and in combination with --bam_list_file)  
* `--bam_list_file`, `-bamf`    A file with all complete paths to the bam files (each new path is a new line)  
* `--out_dir`, `-o` The path to the output directory for the results (if only one sample is predicted, the results will be in this directory, when multiple samples are predicted, directories per sample will be created)  
* `--verbose`, `-v` Write information about the process to the stdout  

**Note:** the sample_name is retrieved from the bam file (sample_name.bam). All bam files must be indexed (.bam.bai or .bai) to be able to be processed.  

#### Result files

Result files are generated per sample, expect for the result_overview files which are only generated when multiple samples are predicted at once.

* `sample_name.result.tsv`  This file contains the prediction of this sample. This includes the raw sample scores for the different models, the interpretation according the defined cut-offs and the results per region per model.
* `region_name.distribution.png`    This are images containing the length distribution of the region in the sample after scalling vs the used MSS samples in training. Gray lines are the used MSS training samples. If 2 out of 3 models predict instability (MSI) for this region, the sample is colored red. If all models pedict stability (MSS) for this region, the sample is colored green. Otherwise the sample is colored yellow.  
* `prediction_plot_*.png` This is similar as the cutoff_matrix plots at the training. The predicted sample is plotted yellow on these plots. These plots help for the interpretation of sample score vs the different cut-offs.  
* `result_overview.tsv`   Only generated for multiple samples. This contains an overview of the outcome of all samples (A combination of all result.tsv files).  
* `result_overview.xslx`  Same file as result_overview.tsv, but in Excel format.  

### Evaluation

This step can be used to evaluate the used method when more samples are being processed. This can help in clinical validation, but also in desciding to create a new model. Evaluation can only be done when IHC information is obtained for all samples.

```bash
python3 -m pipenv run python src/app.py evaluate -m /data/deltamsi/aimodel.deltamsi -ihc /data/all_data.csv -o /data/deltamsi_evaluate -bamf /data/bam_file_paths.txt
```

#### Parameters

* `--model_directory`, `--model`, `-m`  The path to the model directory (the --out_dir in the training step) or the path to the aimodel.deltamsi file (no other files are needed for the prediction).  
* `--ihc_file`, `-ihc`  A text file (tsv or csv) with as first column the sample_name, second the IHC value (pMMR/dMMR, 0/1 or MSS/MSI)  
* `--bam_file`, `-bam`  The bam files of the samples, this can be used multiple times (and in combination with --bam_list_file)  
* `--bam_list_file`, `-bamf`    A file with all complete paths to the bam files (each new path is a new line)  
* `--out_dir`, `-o` The output directory for the model and results  
* `--verbose`, `-v` Write information about the process to the stdout  

**Note:** bam files must have as name: sample_name.bam. All bam files must be indexed (.bam.bai or .bai) to be able to be processed.  

#### Result files

First, the evaluation step will generate exact the same result files as the prediction step. Only one line will be extra included in the result_overview files: the outcome of IHC (True/False). 
 
* `evaluation.txt`    This file contains performance metrics on sample level for the different models.  
* `roc.png`   This is a roc curve, based on the given samples.  

