SERVER_SRC := paramserver/Server.cpp

SERVER_OBJ := $(SERVER_SRC:.cpp=.o)

paramserver/%.o: paramserver/%.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

paramserver/Server: $(THRIFT_OBJ) $(SERVER_OBJ)
	$(CXX) $(LDFLAGS) $(SERVER_OBJ) $(THRIFT_OBJ) -o $@ $(LIBS)

paramserver/clean:
	rm -f paramserver/*.o paramserver/Server

paramserver/all: paramserver/Server
