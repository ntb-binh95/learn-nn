main: main.cpp 
	g++ -o main  main.cpp src/convolution.cpp blas/gemm.c src/mnist.cpp blas/blas.c src/connected.cpp src/network.cpp -Ofast -Iinclude/

conv: src/test_conv.cpp
	g++ -o test src/test_conv.cpp src/convolution.cpp blas/gemm.c blas/blas.c src/mnist.cpp src/network.cpp -O0 -Iinclude/ -g

clean:
	rm -rf main
	rm -rf test
