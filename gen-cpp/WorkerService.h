/**
 * Autogenerated by Thrift Compiler (0.9.1)
 *
 * DO NOT EDIT UNLESS YOU ARE SURE THAT YOU KNOW WHAT YOU ARE DOING
 *  @generated
 */
#ifndef WorkerService_H
#define WorkerService_H

#include <thrift/TDispatchProcessor.h>
#include "distrust_types.h"

namespace distrust {

class WorkerServiceIf {
 public:
  virtual ~WorkerServiceIf() {}
  virtual void heartbeat(HBResponse& _return, const HBRequest& request) = 0;
  virtual void start(const StartRequest& request) = 0;
  virtual void stop() = 0;
  virtual void reassign(const ReassignRequest& request) = 0;
};

class WorkerServiceIfFactory {
 public:
  typedef WorkerServiceIf Handler;

  virtual ~WorkerServiceIfFactory() {}

  virtual WorkerServiceIf* getHandler(const ::apache::thrift::TConnectionInfo& connInfo) = 0;
  virtual void releaseHandler(WorkerServiceIf* /* handler */) = 0;
};

class WorkerServiceIfSingletonFactory : virtual public WorkerServiceIfFactory {
 public:
  WorkerServiceIfSingletonFactory(const boost::shared_ptr<WorkerServiceIf>& iface) : iface_(iface) {}
  virtual ~WorkerServiceIfSingletonFactory() {}

  virtual WorkerServiceIf* getHandler(const ::apache::thrift::TConnectionInfo&) {
    return iface_.get();
  }
  virtual void releaseHandler(WorkerServiceIf* /* handler */) {}

 protected:
  boost::shared_ptr<WorkerServiceIf> iface_;
};

class WorkerServiceNull : virtual public WorkerServiceIf {
 public:
  virtual ~WorkerServiceNull() {}
  void heartbeat(HBResponse& /* _return */, const HBRequest& /* request */) {
    return;
  }
  void start(const StartRequest& /* request */) {
    return;
  }
  void stop() {
    return;
  }
  void reassign(const ReassignRequest& /* request */) {
    return;
  }
};

typedef struct _WorkerService_heartbeat_args__isset {
  _WorkerService_heartbeat_args__isset() : request(false) {}
  bool request;
} _WorkerService_heartbeat_args__isset;

class WorkerService_heartbeat_args {
 public:

  WorkerService_heartbeat_args() {
  }

  virtual ~WorkerService_heartbeat_args() throw() {}

  HBRequest request;

  _WorkerService_heartbeat_args__isset __isset;

  void __set_request(const HBRequest& val) {
    request = val;
  }

  bool operator == (const WorkerService_heartbeat_args & rhs) const
  {
    if (!(request == rhs.request))
      return false;
    return true;
  }
  bool operator != (const WorkerService_heartbeat_args &rhs) const {
    return !(*this == rhs);
  }

  bool operator < (const WorkerService_heartbeat_args & ) const;

  uint32_t read(::apache::thrift::protocol::TProtocol* iprot);
  uint32_t write(::apache::thrift::protocol::TProtocol* oprot) const;

};


class WorkerService_heartbeat_pargs {
 public:


  virtual ~WorkerService_heartbeat_pargs() throw() {}

  const HBRequest* request;

  uint32_t write(::apache::thrift::protocol::TProtocol* oprot) const;

};

typedef struct _WorkerService_heartbeat_result__isset {
  _WorkerService_heartbeat_result__isset() : success(false) {}
  bool success;
} _WorkerService_heartbeat_result__isset;

class WorkerService_heartbeat_result {
 public:

  WorkerService_heartbeat_result() {
  }

  virtual ~WorkerService_heartbeat_result() throw() {}

  HBResponse success;

  _WorkerService_heartbeat_result__isset __isset;

  void __set_success(const HBResponse& val) {
    success = val;
  }

  bool operator == (const WorkerService_heartbeat_result & rhs) const
  {
    if (!(success == rhs.success))
      return false;
    return true;
  }
  bool operator != (const WorkerService_heartbeat_result &rhs) const {
    return !(*this == rhs);
  }

  bool operator < (const WorkerService_heartbeat_result & ) const;

  uint32_t read(::apache::thrift::protocol::TProtocol* iprot);
  uint32_t write(::apache::thrift::protocol::TProtocol* oprot) const;

};

typedef struct _WorkerService_heartbeat_presult__isset {
  _WorkerService_heartbeat_presult__isset() : success(false) {}
  bool success;
} _WorkerService_heartbeat_presult__isset;

class WorkerService_heartbeat_presult {
 public:


  virtual ~WorkerService_heartbeat_presult() throw() {}

  HBResponse* success;

  _WorkerService_heartbeat_presult__isset __isset;

  uint32_t read(::apache::thrift::protocol::TProtocol* iprot);

};

typedef struct _WorkerService_start_args__isset {
  _WorkerService_start_args__isset() : request(false) {}
  bool request;
} _WorkerService_start_args__isset;

class WorkerService_start_args {
 public:

  WorkerService_start_args() {
  }

  virtual ~WorkerService_start_args() throw() {}

  StartRequest request;

  _WorkerService_start_args__isset __isset;

  void __set_request(const StartRequest& val) {
    request = val;
  }

  bool operator == (const WorkerService_start_args & rhs) const
  {
    if (!(request == rhs.request))
      return false;
    return true;
  }
  bool operator != (const WorkerService_start_args &rhs) const {
    return !(*this == rhs);
  }

  bool operator < (const WorkerService_start_args & ) const;

  uint32_t read(::apache::thrift::protocol::TProtocol* iprot);
  uint32_t write(::apache::thrift::protocol::TProtocol* oprot) const;

};


class WorkerService_start_pargs {
 public:


  virtual ~WorkerService_start_pargs() throw() {}

  const StartRequest* request;

  uint32_t write(::apache::thrift::protocol::TProtocol* oprot) const;

};


class WorkerService_start_result {
 public:

  WorkerService_start_result() {
  }

  virtual ~WorkerService_start_result() throw() {}


  bool operator == (const WorkerService_start_result & /* rhs */) const
  {
    return true;
  }
  bool operator != (const WorkerService_start_result &rhs) const {
    return !(*this == rhs);
  }

  bool operator < (const WorkerService_start_result & ) const;

  uint32_t read(::apache::thrift::protocol::TProtocol* iprot);
  uint32_t write(::apache::thrift::protocol::TProtocol* oprot) const;

};


class WorkerService_start_presult {
 public:


  virtual ~WorkerService_start_presult() throw() {}


  uint32_t read(::apache::thrift::protocol::TProtocol* iprot);

};


class WorkerService_stop_args {
 public:

  WorkerService_stop_args() {
  }

  virtual ~WorkerService_stop_args() throw() {}


  bool operator == (const WorkerService_stop_args & /* rhs */) const
  {
    return true;
  }
  bool operator != (const WorkerService_stop_args &rhs) const {
    return !(*this == rhs);
  }

  bool operator < (const WorkerService_stop_args & ) const;

  uint32_t read(::apache::thrift::protocol::TProtocol* iprot);
  uint32_t write(::apache::thrift::protocol::TProtocol* oprot) const;

};


class WorkerService_stop_pargs {
 public:


  virtual ~WorkerService_stop_pargs() throw() {}


  uint32_t write(::apache::thrift::protocol::TProtocol* oprot) const;

};


class WorkerService_stop_result {
 public:

  WorkerService_stop_result() {
  }

  virtual ~WorkerService_stop_result() throw() {}


  bool operator == (const WorkerService_stop_result & /* rhs */) const
  {
    return true;
  }
  bool operator != (const WorkerService_stop_result &rhs) const {
    return !(*this == rhs);
  }

  bool operator < (const WorkerService_stop_result & ) const;

  uint32_t read(::apache::thrift::protocol::TProtocol* iprot);
  uint32_t write(::apache::thrift::protocol::TProtocol* oprot) const;

};


class WorkerService_stop_presult {
 public:


  virtual ~WorkerService_stop_presult() throw() {}


  uint32_t read(::apache::thrift::protocol::TProtocol* iprot);

};

typedef struct _WorkerService_reassign_args__isset {
  _WorkerService_reassign_args__isset() : request(false) {}
  bool request;
} _WorkerService_reassign_args__isset;

class WorkerService_reassign_args {
 public:

  WorkerService_reassign_args() {
  }

  virtual ~WorkerService_reassign_args() throw() {}

  ReassignRequest request;

  _WorkerService_reassign_args__isset __isset;

  void __set_request(const ReassignRequest& val) {
    request = val;
  }

  bool operator == (const WorkerService_reassign_args & rhs) const
  {
    if (!(request == rhs.request))
      return false;
    return true;
  }
  bool operator != (const WorkerService_reassign_args &rhs) const {
    return !(*this == rhs);
  }

  bool operator < (const WorkerService_reassign_args & ) const;

  uint32_t read(::apache::thrift::protocol::TProtocol* iprot);
  uint32_t write(::apache::thrift::protocol::TProtocol* oprot) const;

};


class WorkerService_reassign_pargs {
 public:


  virtual ~WorkerService_reassign_pargs() throw() {}

  const ReassignRequest* request;

  uint32_t write(::apache::thrift::protocol::TProtocol* oprot) const;

};


class WorkerService_reassign_result {
 public:

  WorkerService_reassign_result() {
  }

  virtual ~WorkerService_reassign_result() throw() {}


  bool operator == (const WorkerService_reassign_result & /* rhs */) const
  {
    return true;
  }
  bool operator != (const WorkerService_reassign_result &rhs) const {
    return !(*this == rhs);
  }

  bool operator < (const WorkerService_reassign_result & ) const;

  uint32_t read(::apache::thrift::protocol::TProtocol* iprot);
  uint32_t write(::apache::thrift::protocol::TProtocol* oprot) const;

};


class WorkerService_reassign_presult {
 public:


  virtual ~WorkerService_reassign_presult() throw() {}


  uint32_t read(::apache::thrift::protocol::TProtocol* iprot);

};

class WorkerServiceClient : virtual public WorkerServiceIf {
 public:
  WorkerServiceClient(boost::shared_ptr< ::apache::thrift::protocol::TProtocol> prot) :
    piprot_(prot),
    poprot_(prot) {
    iprot_ = prot.get();
    oprot_ = prot.get();
  }
  WorkerServiceClient(boost::shared_ptr< ::apache::thrift::protocol::TProtocol> iprot, boost::shared_ptr< ::apache::thrift::protocol::TProtocol> oprot) :
    piprot_(iprot),
    poprot_(oprot) {
    iprot_ = iprot.get();
    oprot_ = oprot.get();
  }
  boost::shared_ptr< ::apache::thrift::protocol::TProtocol> getInputProtocol() {
    return piprot_;
  }
  boost::shared_ptr< ::apache::thrift::protocol::TProtocol> getOutputProtocol() {
    return poprot_;
  }
  void heartbeat(HBResponse& _return, const HBRequest& request);
  void send_heartbeat(const HBRequest& request);
  void recv_heartbeat(HBResponse& _return);
  void start(const StartRequest& request);
  void send_start(const StartRequest& request);
  void recv_start();
  void stop();
  void send_stop();
  void recv_stop();
  void reassign(const ReassignRequest& request);
  void send_reassign(const ReassignRequest& request);
  void recv_reassign();
 protected:
  boost::shared_ptr< ::apache::thrift::protocol::TProtocol> piprot_;
  boost::shared_ptr< ::apache::thrift::protocol::TProtocol> poprot_;
  ::apache::thrift::protocol::TProtocol* iprot_;
  ::apache::thrift::protocol::TProtocol* oprot_;
};

class WorkerServiceProcessor : public ::apache::thrift::TDispatchProcessor {
 protected:
  boost::shared_ptr<WorkerServiceIf> iface_;
  virtual bool dispatchCall(::apache::thrift::protocol::TProtocol* iprot, ::apache::thrift::protocol::TProtocol* oprot, const std::string& fname, int32_t seqid, void* callContext);
 private:
  typedef  void (WorkerServiceProcessor::*ProcessFunction)(int32_t, ::apache::thrift::protocol::TProtocol*, ::apache::thrift::protocol::TProtocol*, void*);
  typedef std::map<std::string, ProcessFunction> ProcessMap;
  ProcessMap processMap_;
  void process_heartbeat(int32_t seqid, ::apache::thrift::protocol::TProtocol* iprot, ::apache::thrift::protocol::TProtocol* oprot, void* callContext);
  void process_start(int32_t seqid, ::apache::thrift::protocol::TProtocol* iprot, ::apache::thrift::protocol::TProtocol* oprot, void* callContext);
  void process_stop(int32_t seqid, ::apache::thrift::protocol::TProtocol* iprot, ::apache::thrift::protocol::TProtocol* oprot, void* callContext);
  void process_reassign(int32_t seqid, ::apache::thrift::protocol::TProtocol* iprot, ::apache::thrift::protocol::TProtocol* oprot, void* callContext);
 public:
  WorkerServiceProcessor(boost::shared_ptr<WorkerServiceIf> iface) :
    iface_(iface) {
    processMap_["heartbeat"] = &WorkerServiceProcessor::process_heartbeat;
    processMap_["start"] = &WorkerServiceProcessor::process_start;
    processMap_["stop"] = &WorkerServiceProcessor::process_stop;
    processMap_["reassign"] = &WorkerServiceProcessor::process_reassign;
  }

  virtual ~WorkerServiceProcessor() {}
};

class WorkerServiceProcessorFactory : public ::apache::thrift::TProcessorFactory {
 public:
  WorkerServiceProcessorFactory(const ::boost::shared_ptr< WorkerServiceIfFactory >& handlerFactory) :
      handlerFactory_(handlerFactory) {}

  ::boost::shared_ptr< ::apache::thrift::TProcessor > getProcessor(const ::apache::thrift::TConnectionInfo& connInfo);

 protected:
  ::boost::shared_ptr< WorkerServiceIfFactory > handlerFactory_;
};

class WorkerServiceMultiface : virtual public WorkerServiceIf {
 public:
  WorkerServiceMultiface(std::vector<boost::shared_ptr<WorkerServiceIf> >& ifaces) : ifaces_(ifaces) {
  }
  virtual ~WorkerServiceMultiface() {}
 protected:
  std::vector<boost::shared_ptr<WorkerServiceIf> > ifaces_;
  WorkerServiceMultiface() {}
  void add(boost::shared_ptr<WorkerServiceIf> iface) {
    ifaces_.push_back(iface);
  }
 public:
  void heartbeat(HBResponse& _return, const HBRequest& request) {
    size_t sz = ifaces_.size();
    size_t i = 0;
    for (; i < (sz - 1); ++i) {
      ifaces_[i]->heartbeat(_return, request);
    }
    ifaces_[i]->heartbeat(_return, request);
    return;
  }

  void start(const StartRequest& request) {
    size_t sz = ifaces_.size();
    size_t i = 0;
    for (; i < (sz - 1); ++i) {
      ifaces_[i]->start(request);
    }
    ifaces_[i]->start(request);
  }

  void stop() {
    size_t sz = ifaces_.size();
    size_t i = 0;
    for (; i < (sz - 1); ++i) {
      ifaces_[i]->stop();
    }
    ifaces_[i]->stop();
  }

  void reassign(const ReassignRequest& request) {
    size_t sz = ifaces_.size();
    size_t i = 0;
    for (; i < (sz - 1); ++i) {
      ifaces_[i]->reassign(request);
    }
    ifaces_[i]->reassign(request);
  }

};

} // namespace

#endif
