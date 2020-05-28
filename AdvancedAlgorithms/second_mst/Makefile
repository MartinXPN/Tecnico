all:
	mkdir -p cmake-build
	cd ./cmake-build && \
		cmake .. -DCMAKE_BUILD_TYPE=Release && \
		make && \
		mv MST ..

clean:
	rm MST && rm -rf cmake-build
#		cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=/usr/local/bin/gcc-9 && \
