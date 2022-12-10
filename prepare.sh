#!/bin/bash

# files list
DATASET_NAME="2022-cs4622-lab02"
FILES=( "train.csv" "test.csv" )

# download the dataset from kaggle
if ! [ -x "$(command -v kaggle)" ]; then
  echo 'kaggle is not installed.' >&2
  printf "Do you want to install kaggle? [y/n] "
  read -r answer
  if [ "$answer" = "y" ]; then
    pip install kaggle
    echo "Please setup your kaggle account by running 'kaggle config set -n path -v /path/to/kaggle.json'"
  fi
  exit 0
fi

# create dataset directory if not exist
mkdir -p dataset
cd dataset || exit 1

# if FILES are exist, skip download
for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "dataset/$file exist, skip download"
        exit 0
    fi
done

# if files are already downloaded, skip
if [ ! -f "$DATASET_NAME.zip" ]; then
    kaggle competitions download -c "$DATASET_NAME"
fi

unzip "$DATASET_NAME.zip"

printf "Do you want to delete the zip file? [Y/n] "
read -r answer
if [ "$answer" == "Y" ]; then
    rm "$DATASET_NAME.zip"
fi