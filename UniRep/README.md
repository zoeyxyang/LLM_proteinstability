

# Tape_UniRep.py Readme

The `tape_unirep.py` file is a Python script that downloads the TAPE dataset from lmdb and uses either the 64 or 1900 unit model to fine-tune it. 

## Requirements

To run `tape_unirep.py`, you will need the following Python packages installed:

- torch
- lmdb
- numpy
- scipy
- scikit-learn
- tensorflow

## Models

The `64_weights` and `1900_weights` folders contain the weights for the 64 and 1900 unit variants of UniRep, respectively. The models are saved every three epochs.

## Datasets

The `formatted_new.txt` file contains the processed training dataset, while `formatted_new_test.txt` contains the processed test dataset.

## Unirep.py

The `unirep.py` file contains all the modules needed to get UniRep working.

## Datautils

The `datautils` folder houses different utility functions.

