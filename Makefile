INCDIRS = -I/usr/local/include/thrift \
		  -I..

CXXFLAGS := -Wall \
			-Wextra \
			-Wno-unused-parameter \
			-Wno-unused-variable \
   			-O2 \
   			-g \
   			-pthread \
   			-std=c++11 \
   			$(INCDIRS)

LDFLAGS := -L/usr/local/lib -L../logcabin/build

LIBS := -lboost_system \
		-lboost_filesystem \
		-lthrift \
		-llogcabin \
		-lprotobuf \
		-lpthread

THRIFT_SRC := \
	gen-cpp/WorkerService.cpp \
	gen-cpp/ParamService.cpp \
	gen-cpp/distrust_constants.cpp \
	gen-cpp/distrust_types.cpp

THRIFT_OBJ := $(THRIFT_SRC:.cpp=.o)

default: all

gen-cpp/%.o: gen-cpp/%.cpp
	$(CXX) $(CXXFLAGS) -c -o $@ $<

include worker/Makefile
include paramserver/Makefile

all: worker/all paramserver/all

clean: worker/clean paramserver/clean
	rm -f gen-cpp/*.o
	rm -rf bin
