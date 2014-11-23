CXXFLAGS := -Wall -Wextra -Wno-unused-parameter -Wno-unused-variable \
  -I/usr/local/include/thrift -std=c++11

LDFLAGS := -L/usr/local/lib -lthrift

default: all

include worker/Makefile

all: worker/Worker worker/WorkerClientExample

clean:
	rm -f worker/*.o
	rm -f worker/Worker
	rm -f worker/WorkerClientExample
	rm -f gen-cpp/*.o
