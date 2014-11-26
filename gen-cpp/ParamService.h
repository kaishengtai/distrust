/**
 * Autogenerated by Thrift Compiler (0.9.2)
 *
 * DO NOT EDIT UNLESS YOU ARE SURE THAT YOU KNOW WHAT YOU ARE DOING
 *  @generated
 */
#ifndef ParamService_H
#define ParamService_H

#include <thrift/TDispatchProcessor.h>
#include "distrust_types.h"

namespace distrust {

class ParamServiceIf {
 public:
  virtual ~ParamServiceIf() {}
  virtual void announce(AnnounceResponse& _return, const int32_t worker_port) = 0;
  virtual void push_update(const Params& params) = 0;
  virtual void pull_params(Params& _return) = 0;
};

class ParamServiceIfFactory {
 public:
  typedef ParamServiceIf Handler;

  virtual ~ParamServiceIfFactory() {}

  virtual ParamServiceIf* getHandler(const ::apache::thrift::TConnectionInfo& connInfo) = 0;
  virtual void releaseHandler(ParamServiceIf* /* handler */) = 0;
};

class ParamServiceIfSingletonFactory : virtual public ParamServiceIfFactory {
 public:
  ParamServiceIfSingletonFactory(const boost::shared_ptr<ParamServiceIf>& iface) : iface_(iface) {}
  virtual ~ParamServiceIfSingletonFactory() {}

  virtual ParamServiceIf* getHandler(const ::apache::thrift::TConnectionInfo&) {
    return iface_.get();
  }
  virtual void releaseHandler(ParamServiceIf* /* handler */) {}

 protected:
  boost::shared_ptr<ParamServiceIf> iface_;
};

class ParamServiceNull : virtual public ParamServiceIf {
 public:
  virtual ~ParamServiceNull() {}
  void announce(AnnounceResponse& /* _return */, const int32_t /* worker_port */) {
    return;
  }
  void push_update(const Params& /* params */) {
    return;
  }
  void pull_params(Params& /* _return */) {
    return;
  }
};

typedef struct _ParamService_announce_args__isset {
  _ParamService_announce_args__isset() : worker_port(false) {}
  bool worker_port :1;
} _ParamService_announce_args__isset;

class ParamService_announce_args {
 public:

  static const char* ascii_fingerprint; // = "E86CACEB22240450EDCBEFC3A83970E4";
  static const uint8_t binary_fingerprint[16]; // = {0xE8,0x6C,0xAC,0xEB,0x22,0x24,0x04,0x50,0xED,0xCB,0xEF,0xC3,0xA8,0x39,0x70,0xE4};

  ParamService_announce_args(const ParamService_announce_args&);
  ParamService_announce_args& operator=(const ParamService_announce_args&);
  ParamService_announce_args() : worker_port(0) {
  }

  virtual ~ParamService_announce_args() throw();
  int32_t worker_port;

  _ParamService_announce_args__isset __isset;

  void __set_worker_port(const int32_t val);

  bool operator == (const ParamService_announce_args & rhs) const
  {
    if (!(worker_port == rhs.worker_port))
      return false;
    return true;
  }
  bool operator != (const ParamService_announce_args &rhs) const {
    return !(*this == rhs);
  }

  bool operator < (const ParamService_announce_args & ) const;

  uint32_t read(::apache::thrift::protocol::TProtocol* iprot);
  uint32_t write(::apache::thrift::protocol::TProtocol* oprot) const;

  friend std::ostream& operator<<(std::ostream& out, const ParamService_announce_args& obj);
};


class ParamService_announce_pargs {
 public:

  static const char* ascii_fingerprint; // = "E86CACEB22240450EDCBEFC3A83970E4";
  static const uint8_t binary_fingerprint[16]; // = {0xE8,0x6C,0xAC,0xEB,0x22,0x24,0x04,0x50,0xED,0xCB,0xEF,0xC3,0xA8,0x39,0x70,0xE4};


  virtual ~ParamService_announce_pargs() throw();
  const int32_t* worker_port;

  uint32_t write(::apache::thrift::protocol::TProtocol* oprot) const;

  friend std::ostream& operator<<(std::ostream& out, const ParamService_announce_pargs& obj);
};

typedef struct _ParamService_announce_result__isset {
  _ParamService_announce_result__isset() : success(false) {}
  bool success :1;
} _ParamService_announce_result__isset;

class ParamService_announce_result {
 public:

  static const char* ascii_fingerprint; // = "F787BDE8708ABE5DA500A3E240A0CCC0";
  static const uint8_t binary_fingerprint[16]; // = {0xF7,0x87,0xBD,0xE8,0x70,0x8A,0xBE,0x5D,0xA5,0x00,0xA3,0xE2,0x40,0xA0,0xCC,0xC0};

  ParamService_announce_result(const ParamService_announce_result&);
  ParamService_announce_result& operator=(const ParamService_announce_result&);
  ParamService_announce_result() {
  }

  virtual ~ParamService_announce_result() throw();
  AnnounceResponse success;

  _ParamService_announce_result__isset __isset;

  void __set_success(const AnnounceResponse& val);

  bool operator == (const ParamService_announce_result & rhs) const
  {
    if (!(success == rhs.success))
      return false;
    return true;
  }
  bool operator != (const ParamService_announce_result &rhs) const {
    return !(*this == rhs);
  }

  bool operator < (const ParamService_announce_result & ) const;

  uint32_t read(::apache::thrift::protocol::TProtocol* iprot);
  uint32_t write(::apache::thrift::protocol::TProtocol* oprot) const;

  friend std::ostream& operator<<(std::ostream& out, const ParamService_announce_result& obj);
};

typedef struct _ParamService_announce_presult__isset {
  _ParamService_announce_presult__isset() : success(false) {}
  bool success :1;
} _ParamService_announce_presult__isset;

class ParamService_announce_presult {
 public:

  static const char* ascii_fingerprint; // = "F787BDE8708ABE5DA500A3E240A0CCC0";
  static const uint8_t binary_fingerprint[16]; // = {0xF7,0x87,0xBD,0xE8,0x70,0x8A,0xBE,0x5D,0xA5,0x00,0xA3,0xE2,0x40,0xA0,0xCC,0xC0};


  virtual ~ParamService_announce_presult() throw();
  AnnounceResponse* success;

  _ParamService_announce_presult__isset __isset;

  uint32_t read(::apache::thrift::protocol::TProtocol* iprot);

  friend std::ostream& operator<<(std::ostream& out, const ParamService_announce_presult& obj);
};

typedef struct _ParamService_push_update_args__isset {
  _ParamService_push_update_args__isset() : params(false) {}
  bool params :1;
} _ParamService_push_update_args__isset;

class ParamService_push_update_args {
 public:

  static const char* ascii_fingerprint; // = "3DADC0651ECCD00EE4963E997F9778AB";
  static const uint8_t binary_fingerprint[16]; // = {0x3D,0xAD,0xC0,0x65,0x1E,0xCC,0xD0,0x0E,0xE4,0x96,0x3E,0x99,0x7F,0x97,0x78,0xAB};

  ParamService_push_update_args(const ParamService_push_update_args&);
  ParamService_push_update_args& operator=(const ParamService_push_update_args&);
  ParamService_push_update_args() {
  }

  virtual ~ParamService_push_update_args() throw();
  Params params;

  _ParamService_push_update_args__isset __isset;

  void __set_params(const Params& val);

  bool operator == (const ParamService_push_update_args & rhs) const
  {
    if (!(params == rhs.params))
      return false;
    return true;
  }
  bool operator != (const ParamService_push_update_args &rhs) const {
    return !(*this == rhs);
  }

  bool operator < (const ParamService_push_update_args & ) const;

  uint32_t read(::apache::thrift::protocol::TProtocol* iprot);
  uint32_t write(::apache::thrift::protocol::TProtocol* oprot) const;

  friend std::ostream& operator<<(std::ostream& out, const ParamService_push_update_args& obj);
};


class ParamService_push_update_pargs {
 public:

  static const char* ascii_fingerprint; // = "3DADC0651ECCD00EE4963E997F9778AB";
  static const uint8_t binary_fingerprint[16]; // = {0x3D,0xAD,0xC0,0x65,0x1E,0xCC,0xD0,0x0E,0xE4,0x96,0x3E,0x99,0x7F,0x97,0x78,0xAB};


  virtual ~ParamService_push_update_pargs() throw();
  const Params* params;

  uint32_t write(::apache::thrift::protocol::TProtocol* oprot) const;

  friend std::ostream& operator<<(std::ostream& out, const ParamService_push_update_pargs& obj);
};


class ParamService_push_update_result {
 public:

  static const char* ascii_fingerprint; // = "99914B932BD37A50B983C5E7C90AE93B";
  static const uint8_t binary_fingerprint[16]; // = {0x99,0x91,0x4B,0x93,0x2B,0xD3,0x7A,0x50,0xB9,0x83,0xC5,0xE7,0xC9,0x0A,0xE9,0x3B};

  ParamService_push_update_result(const ParamService_push_update_result&);
  ParamService_push_update_result& operator=(const ParamService_push_update_result&);
  ParamService_push_update_result() {
  }

  virtual ~ParamService_push_update_result() throw();

  bool operator == (const ParamService_push_update_result & /* rhs */) const
  {
    return true;
  }
  bool operator != (const ParamService_push_update_result &rhs) const {
    return !(*this == rhs);
  }

  bool operator < (const ParamService_push_update_result & ) const;

  uint32_t read(::apache::thrift::protocol::TProtocol* iprot);
  uint32_t write(::apache::thrift::protocol::TProtocol* oprot) const;

  friend std::ostream& operator<<(std::ostream& out, const ParamService_push_update_result& obj);
};


class ParamService_push_update_presult {
 public:

  static const char* ascii_fingerprint; // = "99914B932BD37A50B983C5E7C90AE93B";
  static const uint8_t binary_fingerprint[16]; // = {0x99,0x91,0x4B,0x93,0x2B,0xD3,0x7A,0x50,0xB9,0x83,0xC5,0xE7,0xC9,0x0A,0xE9,0x3B};


  virtual ~ParamService_push_update_presult() throw();

  uint32_t read(::apache::thrift::protocol::TProtocol* iprot);

  friend std::ostream& operator<<(std::ostream& out, const ParamService_push_update_presult& obj);
};


class ParamService_pull_params_args {
 public:

  static const char* ascii_fingerprint; // = "99914B932BD37A50B983C5E7C90AE93B";
  static const uint8_t binary_fingerprint[16]; // = {0x99,0x91,0x4B,0x93,0x2B,0xD3,0x7A,0x50,0xB9,0x83,0xC5,0xE7,0xC9,0x0A,0xE9,0x3B};

  ParamService_pull_params_args(const ParamService_pull_params_args&);
  ParamService_pull_params_args& operator=(const ParamService_pull_params_args&);
  ParamService_pull_params_args() {
  }

  virtual ~ParamService_pull_params_args() throw();

  bool operator == (const ParamService_pull_params_args & /* rhs */) const
  {
    return true;
  }
  bool operator != (const ParamService_pull_params_args &rhs) const {
    return !(*this == rhs);
  }

  bool operator < (const ParamService_pull_params_args & ) const;

  uint32_t read(::apache::thrift::protocol::TProtocol* iprot);
  uint32_t write(::apache::thrift::protocol::TProtocol* oprot) const;

  friend std::ostream& operator<<(std::ostream& out, const ParamService_pull_params_args& obj);
};


class ParamService_pull_params_pargs {
 public:

  static const char* ascii_fingerprint; // = "99914B932BD37A50B983C5E7C90AE93B";
  static const uint8_t binary_fingerprint[16]; // = {0x99,0x91,0x4B,0x93,0x2B,0xD3,0x7A,0x50,0xB9,0x83,0xC5,0xE7,0xC9,0x0A,0xE9,0x3B};


  virtual ~ParamService_pull_params_pargs() throw();

  uint32_t write(::apache::thrift::protocol::TProtocol* oprot) const;

  friend std::ostream& operator<<(std::ostream& out, const ParamService_pull_params_pargs& obj);
};

typedef struct _ParamService_pull_params_result__isset {
  _ParamService_pull_params_result__isset() : success(false) {}
  bool success :1;
} _ParamService_pull_params_result__isset;

class ParamService_pull_params_result {
 public:

  static const char* ascii_fingerprint; // = "8F16AB9F450B00151E492FEF8ECCA0A6";
  static const uint8_t binary_fingerprint[16]; // = {0x8F,0x16,0xAB,0x9F,0x45,0x0B,0x00,0x15,0x1E,0x49,0x2F,0xEF,0x8E,0xCC,0xA0,0xA6};

  ParamService_pull_params_result(const ParamService_pull_params_result&);
  ParamService_pull_params_result& operator=(const ParamService_pull_params_result&);
  ParamService_pull_params_result() {
  }

  virtual ~ParamService_pull_params_result() throw();
  Params success;

  _ParamService_pull_params_result__isset __isset;

  void __set_success(const Params& val);

  bool operator == (const ParamService_pull_params_result & rhs) const
  {
    if (!(success == rhs.success))
      return false;
    return true;
  }
  bool operator != (const ParamService_pull_params_result &rhs) const {
    return !(*this == rhs);
  }

  bool operator < (const ParamService_pull_params_result & ) const;

  uint32_t read(::apache::thrift::protocol::TProtocol* iprot);
  uint32_t write(::apache::thrift::protocol::TProtocol* oprot) const;

  friend std::ostream& operator<<(std::ostream& out, const ParamService_pull_params_result& obj);
};

typedef struct _ParamService_pull_params_presult__isset {
  _ParamService_pull_params_presult__isset() : success(false) {}
  bool success :1;
} _ParamService_pull_params_presult__isset;

class ParamService_pull_params_presult {
 public:

  static const char* ascii_fingerprint; // = "8F16AB9F450B00151E492FEF8ECCA0A6";
  static const uint8_t binary_fingerprint[16]; // = {0x8F,0x16,0xAB,0x9F,0x45,0x0B,0x00,0x15,0x1E,0x49,0x2F,0xEF,0x8E,0xCC,0xA0,0xA6};


  virtual ~ParamService_pull_params_presult() throw();
  Params* success;

  _ParamService_pull_params_presult__isset __isset;

  uint32_t read(::apache::thrift::protocol::TProtocol* iprot);

  friend std::ostream& operator<<(std::ostream& out, const ParamService_pull_params_presult& obj);
};

class ParamServiceClient : virtual public ParamServiceIf {
 public:
  ParamServiceClient(boost::shared_ptr< ::apache::thrift::protocol::TProtocol> prot) {
    setProtocol(prot);
  }
  ParamServiceClient(boost::shared_ptr< ::apache::thrift::protocol::TProtocol> iprot, boost::shared_ptr< ::apache::thrift::protocol::TProtocol> oprot) {
    setProtocol(iprot,oprot);
  }
 private:
  void setProtocol(boost::shared_ptr< ::apache::thrift::protocol::TProtocol> prot) {
  setProtocol(prot,prot);
  }
  void setProtocol(boost::shared_ptr< ::apache::thrift::protocol::TProtocol> iprot, boost::shared_ptr< ::apache::thrift::protocol::TProtocol> oprot) {
    piprot_=iprot;
    poprot_=oprot;
    iprot_ = iprot.get();
    oprot_ = oprot.get();
  }
 public:
  boost::shared_ptr< ::apache::thrift::protocol::TProtocol> getInputProtocol() {
    return piprot_;
  }
  boost::shared_ptr< ::apache::thrift::protocol::TProtocol> getOutputProtocol() {
    return poprot_;
  }
  void announce(AnnounceResponse& _return, const int32_t worker_port);
  void send_announce(const int32_t worker_port);
  void recv_announce(AnnounceResponse& _return);
  void push_update(const Params& params);
  void send_push_update(const Params& params);
  void recv_push_update();
  void pull_params(Params& _return);
  void send_pull_params();
  void recv_pull_params(Params& _return);
 protected:
  boost::shared_ptr< ::apache::thrift::protocol::TProtocol> piprot_;
  boost::shared_ptr< ::apache::thrift::protocol::TProtocol> poprot_;
  ::apache::thrift::protocol::TProtocol* iprot_;
  ::apache::thrift::protocol::TProtocol* oprot_;
};

class ParamServiceProcessor : public ::apache::thrift::TDispatchProcessor {
 protected:
  boost::shared_ptr<ParamServiceIf> iface_;
  virtual bool dispatchCall(::apache::thrift::protocol::TProtocol* iprot, ::apache::thrift::protocol::TProtocol* oprot, const std::string& fname, int32_t seqid, void* callContext);
 private:
  typedef  void (ParamServiceProcessor::*ProcessFunction)(int32_t, ::apache::thrift::protocol::TProtocol*, ::apache::thrift::protocol::TProtocol*, void*);
  typedef std::map<std::string, ProcessFunction> ProcessMap;
  ProcessMap processMap_;
  void process_announce(int32_t seqid, ::apache::thrift::protocol::TProtocol* iprot, ::apache::thrift::protocol::TProtocol* oprot, void* callContext);
  void process_push_update(int32_t seqid, ::apache::thrift::protocol::TProtocol* iprot, ::apache::thrift::protocol::TProtocol* oprot, void* callContext);
  void process_pull_params(int32_t seqid, ::apache::thrift::protocol::TProtocol* iprot, ::apache::thrift::protocol::TProtocol* oprot, void* callContext);
 public:
  ParamServiceProcessor(boost::shared_ptr<ParamServiceIf> iface) :
    iface_(iface) {
    processMap_["announce"] = &ParamServiceProcessor::process_announce;
    processMap_["push_update"] = &ParamServiceProcessor::process_push_update;
    processMap_["pull_params"] = &ParamServiceProcessor::process_pull_params;
  }

  virtual ~ParamServiceProcessor() {}
};

class ParamServiceProcessorFactory : public ::apache::thrift::TProcessorFactory {
 public:
  ParamServiceProcessorFactory(const ::boost::shared_ptr< ParamServiceIfFactory >& handlerFactory) :
      handlerFactory_(handlerFactory) {}

  ::boost::shared_ptr< ::apache::thrift::TProcessor > getProcessor(const ::apache::thrift::TConnectionInfo& connInfo);

 protected:
  ::boost::shared_ptr< ParamServiceIfFactory > handlerFactory_;
};

class ParamServiceMultiface : virtual public ParamServiceIf {
 public:
  ParamServiceMultiface(std::vector<boost::shared_ptr<ParamServiceIf> >& ifaces) : ifaces_(ifaces) {
  }
  virtual ~ParamServiceMultiface() {}
 protected:
  std::vector<boost::shared_ptr<ParamServiceIf> > ifaces_;
  ParamServiceMultiface() {}
  void add(boost::shared_ptr<ParamServiceIf> iface) {
    ifaces_.push_back(iface);
  }
 public:
  void announce(AnnounceResponse& _return, const int32_t worker_port) {
    size_t sz = ifaces_.size();
    size_t i = 0;
    for (; i < (sz - 1); ++i) {
      ifaces_[i]->announce(_return, worker_port);
    }
    ifaces_[i]->announce(_return, worker_port);
    return;
  }

  void push_update(const Params& params) {
    size_t sz = ifaces_.size();
    size_t i = 0;
    for (; i < (sz - 1); ++i) {
      ifaces_[i]->push_update(params);
    }
    ifaces_[i]->push_update(params);
  }

  void pull_params(Params& _return) {
    size_t sz = ifaces_.size();
    size_t i = 0;
    for (; i < (sz - 1); ++i) {
      ifaces_[i]->pull_params(_return);
    }
    ifaces_[i]->pull_params(_return);
    return;
  }

};

} // namespace

#endif
