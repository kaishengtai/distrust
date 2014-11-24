#include "distrust/gen-cpp/ParamService.h"
#include "logcabin/Client/Client.h"
#include <getopt.h>
#include <thrift/protocol/TBinaryProtocol.h>
#include <thrift/server/TSimpleServer.h>
#include <thrift/transport/TServerSocket.h>
#include <thrift/transport/TBufferTransports.h>

using namespace ::apache::thrift;
using namespace ::apache::thrift::protocol;
using namespace ::apache::thrift::transport;
using namespace ::apache::thrift::server;

using LogCabin::Client::Cluster;
using LogCabin::Client::Tree;

using boost::shared_ptr;

using namespace  ::distrust;

class ParamServiceHandler : virtual public ParamServiceIf {
 public:
  ParamServiceHandler(const std::string &logcabin_cluster)
      : cluster_(logcabin_cluster) {
  }

  void announce(AnnounceResponse& _return, const AnnounceRequest& request) {
    printf("announce\n");
  }

  void push_update(UpdateResponse& _return, const UpdateRequest& request) {
    printf("push_update\n");
  }

  void pull_params(PullResponse& _return, const PullRequest& request) {
    printf("pull_params\n");
  }

 private:
  Cluster cluster_;
};

/**
 * Parses argv for the main function.
 */
class OptionParser {
 public:
  OptionParser(int& argc, char**& argv)
      : argc_(argc),
        argv_(argv),
        cluster("logcabin:61023"),
        port(0) {
    static struct option longOptions[] = {
      {"cluster",  required_argument, NULL, 'c'},
      {"port", required_argument, 0, 'p'},
      {"help",  no_argument, NULL, 'h'},
      {0, 0, 0, 0}
    };

    int c;
    while ((c = getopt_long(argc, argv, "c:p:h", longOptions, NULL)) != -1) {
      switch (c) {
        case 'c':
          cluster = optarg;
          break;
        case 'p':
          port = atoi(optarg);
        case 'h':
          usage();
          exit(0);
        case '?':
        default:
          usage();
          exit(1);
      }
    }
  }

  void usage() {
      std::cout << "Usage: " << argv_[0] << " [options]"
                << std::endl;
      std::cout << "Options: " << std::endl;
      std::cout << "  -c, --cluster <address> "
                << "The network address of the LogCabin cluster "
                << "(default: logcabin:61023)" << std::endl;
      std::cout << "  -p, --port <port> "
                << "The port that the ParamServer listens to RPC" << std::endl;
      std::cout << "  -h, --help              "
                << "Print this usage information" << std::endl;
  }

  int& argc_;
  char**& argv_;
  std::string cluster;
  int port;
};


int main(int argc, char **argv) {
  // The following trivial code tests whether LogCabin is up and running.
  Cluster cluster("logcabin:61023");
  Tree tree = cluster.getTree();
  tree.makeDirectoryEx("/etc");
  tree.writeEx("/etc/passwd", "ha");
  std::string contents = tree.readEx("/etc/passwd");
  printf("Contents of logcabin at %s: %s\n", "/etc/passwd", contents.c_str());
  tree.removeDirectoryEx("/etc");

  OptionParser options(argc, argv);
  shared_ptr<ParamServiceHandler> handler(
      new ParamServiceHandler(options.cluster));
  shared_ptr<TProcessor> processor(new ParamServiceProcessor(handler));
  shared_ptr<TServerTransport> serverTransport(new TServerSocket(options.port));
  shared_ptr<TTransportFactory> transportFactory(
    new TBufferedTransportFactory());
  shared_ptr<TProtocolFactory> protocolFactory(new TBinaryProtocolFactory());

  TSimpleServer server(processor, serverTransport, transportFactory,
                       protocolFactory);
  printf("paramserver: using LogCabin cluster: %s\n", options.cluster.c_str());
  printf("paramserver: start serving on port %d\n", options.port);
  server.serve();
  return 0;
}

