WORKER_SRC := worker/Worker.cpp \
			  worker/WorkerServiceHandler.cpp \
			  worker/LanguageModel.cpp

WORKER_OBJ := $(WORKER_SRC:.cpp=.o)

worker/%.o: worker/%.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

Worker: $(WORKER_OBJ) $(THRIFT_OBJ)
	mkdir -p bin
	$(CXX) $(LDFLAGS) $(WORKER_OBJ) $(THRIFT_OBJ) -o bin/$@ $(LIBS)

worker/clean:
	rm -f worker/*.o

worker/all: Worker
