THRIFT_SRC := \
	gen-cpp/WorkerService.cpp \
	gen-cpp/distrust_constants.cpp \
	gen-cpp/distrust_types.cpp

THRIFT_OBJ := $(THRIFT_SRC:.cpp=.o)

WORKER_SRC := worker/Worker.cpp

WORKER_OBJ := $(WORKER_SRC:.cpp=.o)

WORKER_CLIENT_SRC := worker/WorkerClientExample.cpp

WORKER_CLIENT_OBJ := $(WORKER_CLIENT_SRC:.cpp=.o)

gen-cpp/%.o: gen-cpp/%.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

worker/%.o: worker/%.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

worker/Worker: $(WORKER_OBJ) $(THRIFT_OBJ)
	$(CXX) $(LDFLAGS) $(WORKER_OBJ) $(THRIFT_OBJ) -o $@ 

worker/WorkerClientExample: $(WORKER_CLIENT_OBJ) $(THRIFT_OBJ)
	$(CXX) $(LDFLAGS) $(WORKER_CLIENT_OBJ) $(THRIFT_OBJ) -o $@
