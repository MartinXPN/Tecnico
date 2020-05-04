all:
	mkdir -p cmake-build
	cd ./cmake-build && \
		cmake .. -DCMAKE_BUILD_TYPE=Release && \
		make && \
		mv matFact .. && mv matFact-omp .. && mv matFact-mpi ..

clean:
	rm matFact matFact-omp matFact-mpi && rm -rf cmake-build

#		cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=/usr/local/bin/gcc-9 && \