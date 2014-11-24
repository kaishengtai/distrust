WORKER_SRC := worker/Worker.cpp

WORKER_OBJ := $(WORKER_SRC:.cpp=.o)

WORKER_CLIENT_SRC := worker/WorkerClientExample.cpp

WORKER_CLIENT_OBJ := $(WORKER_CLIENT_SRC:.cpp=.o)

worker/%.o: worker/%.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

Worker: $(WORKER_OBJ) $(THRIFT_OBJ)
	mkdir -p bin
	$(CXX) $(LDFLAGS) $(WORKER_OBJ) $(THRIFT_OBJ) -o bin/$@ $(LIBS)

WorkerClientExample: $(WORKER_CLIENT_OBJ) $(THRIFT_OBJ)
	mkdir -p bin
	$(CXX) $(LDFLAGS) $(WORKER_CLIENT_OBJ) $(THRIFT_OBJ) -o bin/$@ $(LIBS)

worker/clean:
	rm -f worker/*.o

worker/all: Worker WorkerClientExample
