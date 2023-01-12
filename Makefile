main: main.cpp 
	g++ -o main  main.cpp src/convolution.cpp blas/gemm.c src/mnist.cpp blas/blas.c src/connected.cpp src/network.cpp -Ofast -Iinclude/

clean:
	rm -rf main
