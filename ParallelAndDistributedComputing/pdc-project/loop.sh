#! /bin/sh
# change above to your shell.\
echo Specify path to instance you want to run:
read rel_path
for i in 2 4 8 16
do 
	OMP_NUM_THREADS=$i
	./matFact-omp $rel_path
done
