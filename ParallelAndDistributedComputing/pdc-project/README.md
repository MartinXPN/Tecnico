# pdc-project

### Instruction to run the sequential program
```shell script
cd path/to/.../pdc-project
export CXX=/usr/local/bin/g++-9

# Option 1: With cmake
mkdir cmake-build
cmake -B cmake-build                # we want to keep the temporary cmake files here
make -C cmake-build                 # for compilation and getting the executable
cmake-build/matFact instances/inst0.in


# Option 2: Without cmake
CXX matFact.cpp -o matFact
matFact instances/inst0.in


# Option 3: Makefile
make
matFact instances/inst0.in

# Option 4: run.sh -> checks correctness
./run.sh inst0
```


### Instruction to run the OpenMP program
```shell script
cd path/to/.../pdc-project
export OMP_NUM_THREADS=4
export CXX=/usr/local/bin/g++-9

# Option 1: With cmake (completely the same as the sequential, only use `matFact-omp` instead of `matFact`
mkdir cmake-build
cmake -B cmake-build                # we want to keep the temporary cmake files here
make -C cmake-build                 # for compilation and getting the executable
cmake-build/matFact-omp instances/inst0.in


# Option 2: Without cmake
CXX -fopenmp matFact-omp.cpp -o matFact-omp
matFact-omp instances/inst0.in


# Option 3: Makefile
make
matFact-omp instances/inst0.in

# Option 4: run.sh -> checks correctness
./run.sh inst0 -omp
```

### Instruction to run the MPI program
```shell script
cd path/to/.../pdc-project
export CXX=/usr/local/bin/g++-9
export CC=/usr/local/bin/gcc-9
export PMIX_MCA_gds=hash
export OMPI_MCA_btl=self,tcp

# Option 1: With cmake (completely the same as the sequential, only use `matFact-mpi` instead of `matFact`
mkdir cmake-build
cmake -B cmake-build                # we want to keep the temporary cmake files here
make -C cmake-build                 # for compilation and getting the executable
mpirun -np 2 cmake-build/matFact-mpi instances/inst0.in


# Option 2: Without cmake
CXX matFact-mpi.cpp -o matFact-mpi
mpirun -np 2 matFact-mpi instances/inst0.in


# Option 3: Makefile
make
matFact-omp instances/inst0.in
```

### Instruction to run the shell script
```shell script
cd path/to/.../pdc-project
export OMP_NUM_THREADS #export first to source the env variable!!
chmod 755 loop.sh
. ./loop.sh
````
#it will ask you to enter the path to your instance, this should be the same path as above e.g. instances/inst0.in

### Instruction to run the speedup/correctness script
```
python speedup.py [Check correctness] [Number of threads to start with (min 2)] [Max num of threads (min 2)] [Thread num step] [instances]

python speedup.py True 4 8 1 inst0 inst1

python speedup.py False 2 2 1 all

python speedup.py True 8 16 2 allBig
```
