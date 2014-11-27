SERVER_SRC := paramserver/Server.cpp \
		      LanguageModel.cpp

SERVER_OBJ := $(SERVER_SRC:.cpp=.o)

paramserver/%.o: paramserver/%.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

Server: $(THRIFT_OBJ) $(SERVER_OBJ)
	mkdir -p bin
	$(CXX) $(LDFLAGS) $(SERVER_OBJ) $(THRIFT_OBJ) -o bin/$@ $(LIBS)

paramserver/clean:
	rm -f paramserver/*.o

paramserver/all: Server
