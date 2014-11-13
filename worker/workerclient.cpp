/**
 * A simple client for testing.
 */

#include "../gen-cpp/Worker.h"

#include <transport/TSocket.h>
#include <transport/TBufferTransports.h>
#include <protocol/TBinaryProtocol.h>

using namespace apache::thrift;
using namespace apache::thrift::protocol;
using namespace apache::thrift::transport;

using boost::shared_ptr;

using namespace distrust;

int main(int argc, char **argv) {
    shared_ptr<TSocket> socket(new TSocket("localhost", 9090));
    shared_ptr<TTransport> transport(new TBufferedTransport(socket));
    shared_ptr<TProtocol> protocol(new TBinaryProtocol(transport));

    WorkerClient client(protocol);
    transport->open();
    client.stop();
    transport->close();
    return 0;
}
