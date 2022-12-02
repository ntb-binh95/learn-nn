main: main.cpp gemm.cpp
	g++ -o main main.cpp gemm.cpp mnist.cpp

example: example.cpp gemm.cpp
	g++ -o example example.cpp gemm.cpp

test: test.cpp gemm.cpp
	g++ -o test test.cpp gemm.cpp

clean:
	rm -rf main
	rm -rf example
	rm -rf test
