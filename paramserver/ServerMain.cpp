#include "Server.h"

#include <getopt.h>

#include "logcabin/Client/Client.h"

using LogCabin::Client::Cluster;
using LogCabin::Client::Tree;

/**
 * Parses argv for the main function.
 */
class OptionParser {
 public:
  OptionParser(int& argc, char**& argv)
      : argc_(argc),
        argv_(argv),
        window_size(3),
        wordvec_dim(50),
        hidden_dim(100),
        port(8000),
        cluster("logcabin:61023") {
    static struct option longOptions[] = {
      {"cluster",  required_argument, NULL, 'c'},
      {"port", required_argument, 0, 'p'},
      {"data", required_argument, 0, 'd'},
      {"window", required_argument, 0, 'w'},
      {"wordvec", required_argument, 0, 'v'},
      {"hidden", required_argument, 0, 'H'},
      {"help",  no_argument, NULL, 'h'},
      {0, 0, 0, 0}
    };

    int c;
    while ((c = getopt_long(argc, argv, "c:p:d:w:v:H:h",
                            longOptions, NULL)) != -1) {
      switch (c) {
        case 'c':
          cluster = optarg;
          break;
        case 'p':
          port = atoi(optarg);
          break;
        case 'd':
          data_dir = optarg;
          break;
        case 'w':
          window_size = atoi(optarg);
          break;
        case 'v':
          wordvec_dim = atoi(optarg);
          break;
        case 'H':
          hidden_dim = atoi(optarg);
          break;
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
      std::cout << "  -d, --data <dir> "
                << "The path to the directory that contains the dataset "
                << "(assumed to contain vocabulary in vocab.txt)" << std::endl;
      std::cout << "  -w, --window <dim>" << std::endl;
      std::cout << "  -v, --wordvec <dim>" << std::endl;
      std::cout << "  -H, --hidden <dim>" << std::endl;
      std::cout << "  -h, --help              "
                << "Print this usage information" << std::endl;
  }

  int& argc_;
  char**& argv_;

  int window_size;
  int wordvec_dim;
  int hidden_dim;
  int port;
  std::string cluster;
  std::string data_dir;
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
  
  printf("paramserver: using LogCabin cluster: %s\n", options.cluster.c_str());
  printf("paramserver: start serving on port %d\n", options.port);
  distrust::ParamServer param_server(
    options.window_size,
    options.wordvec_dim,
    options.hidden_dim,
    options.port,
    options.cluster,
    options.data_dir);
  param_server.run();
  return 0;
}
