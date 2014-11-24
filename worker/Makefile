WORKER_SRC := worker/Worker.cpp

WORKER_OBJ := $(WORKER_SRC:.cpp=.o)

WORKER_CLIENT_SRC := worker/WorkerClientExample.cpp

WORKER_CLIENT_OBJ := $(WORKER_CLIENT_SRC:.cpp=.o)

worker/%.o: worker/%.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

worker/Worker: $(WORKER_OBJ) $(THRIFT_OBJ)
	$(CXX) $(LDFLAGS) $(WORKER_OBJ) $(THRIFT_OBJ) -o $@ $(LIBS)

worker/WorkerClientExample: $(WORKER_CLIENT_OBJ) $(THRIFT_OBJ)
	$(CXX) $(LDFLAGS) $(WORKER_CLIENT_OBJ) $(THRIFT_OBJ) -o $@ $(LIBS)

worker/clean:
	rm -f worker/*.o worker/Worker worker/WorkerClientExample

worker/all: worker/Worker worker/WorkerClientExample
