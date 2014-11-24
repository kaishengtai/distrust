INCDIRS = -I/usr/local/include -I../../
CXXFLAGS = -std=c++11 -Wall -O2 -g $(INCDIRS)
LDFLAGS := -L/usr/local/lib -L../../logcabin/build -lthrift -llogcabin -lprotobuf
THRIFT_SRC := \
	../gen-cpp/ParamService.cpp \
	../gen-cpp/distrust_constants.cpp \
	../gen-cpp/distrust_types.cpp

SERVER_SRC := ./server.cpp

server: $(THRIFT_SRC) $(SERVER_SRC)
	$(CXX) $(CXXFLAGS) -o $@ $(SERVER_SRC) $(THRIFT_SRC) $(LDFLAGS)

clean:
	rm -f ./*.o ./server
