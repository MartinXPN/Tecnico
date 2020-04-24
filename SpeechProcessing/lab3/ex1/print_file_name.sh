#!/bin/bash

#########################################################################
#                                                                       #
# Script for printing file names        		                #
# --------------------------------------------------------------------- # 
# Date: April 2020                                                      #
#                                                                       #
#########################################################################

# This script loops over the three partitions (train, dev and test), creates 
# a list with the names of all wav files correspondent to each partition,
# and prints the name of each .wav file. You will have to adapt the script
# to make it perform the task you need, instead printing the names of the
# files.

## Installation configuration # EDIT THIS!
mainDir=`pwd`
wavDir="$mainDir/../corpus/wav" 
labelsDir="$mainDir/../corpus/labels"


## Datasets to process
TRAINSET="train"
DEVSET="dev"
TESTSET="test"
DATASETS=($TRAINSET $DEVSET $TESTSET)


# loops ober the partitions
for dataset in ${DATASETS[*]}; do
	echo "Partition: $dataset set"

	# creates a list of all .wav files. Notice that for the train and dev sets,
	# the list is the same as the labels file, assuring that files are in the
	# same order as in the labels file. For the test set, we create a list of
	# all the .wav files from the wav directory, and save that list in the labels
	# dir for later use.
	if [ "$dataset" == "$TESTSET" ]; then
		find $wavDir/$dataset | grep "\.wav" | awk 'BEGING{print "file_id"} {print $0}' > $labelsDir/${dataset}_file_list.txt
		awk -F, 'NR>1{print $1}' $labelsDir/${dataset}_file_list.txt > file_list.txt
	else
		awk -F, 'NR>1{print $1}' $labelsDir/${dataset}_labels.csv > file_list.txt
	fi

	# loops over the list of files
	while read line; do
		
		# this line prints the file name and the path to an output_file.txt
		# replace it by a soxi command (Part 1.2) or an openSMILE command (Part1.3).
		echo "file name: $line complete path to file: $wavDir/$dataset/$line" >> output_file.txt


		# the following command computes sum and average of all numbers in the first column of 
		# file.txt. Replace file.txt by the file you generated with all recordings' duration.
		#awk 'BEGIN{sum=0} {sum += $1} END{print "Sum: " sum " average: " sum/NR}' file.txt
	done < file_list.txt

  # deletes file_list.txt
  rm file_list.txt

done
