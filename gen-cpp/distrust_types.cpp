/**
 * Autogenerated by Thrift Compiler (0.9.1)
 *
 * DO NOT EDIT UNLESS YOU ARE SURE THAT YOU KNOW WHAT YOU ARE DOING
 *  @generated
 */
#include "distrust_types.h"

#include <algorithm>

namespace distrust {

const char* ModelInfo::ascii_fingerprint = "62CBF95059CB084430B0BABE2E5A68C7";
const uint8_t ModelInfo::binary_fingerprint[16] = {0x62,0xCB,0xF9,0x50,0x59,0xCB,0x08,0x44,0x30,0xB0,0xBA,0xBE,0x2E,0x5A,0x68,0xC7};

uint32_t ModelInfo::read(::apache::thrift::protocol::TProtocol* iprot) {

  uint32_t xfer = 0;
  std::string fname;
  ::apache::thrift::protocol::TType ftype;
  int16_t fid;

  xfer += iprot->readStructBegin(fname);

  using ::apache::thrift::protocol::TProtocolException;


  while (true)
  {
    xfer += iprot->readFieldBegin(fname, ftype, fid);
    if (ftype == ::apache::thrift::protocol::T_STOP) {
      break;
    }
    switch (fid)
    {
      case 1:
        if (ftype == ::apache::thrift::protocol::T_I32) {
          xfer += iprot->readI32(this->window_size);
          this->__isset.window_size = true;
        } else {
          xfer += iprot->skip(ftype);
        }
        break;
      case 2:
        if (ftype == ::apache::thrift::protocol::T_I32) {
          xfer += iprot->readI32(this->vocab_size);
          this->__isset.vocab_size = true;
        } else {
          xfer += iprot->skip(ftype);
        }
        break;
      case 3:
        if (ftype == ::apache::thrift::protocol::T_I32) {
          xfer += iprot->readI32(this->start_token_index);
          this->__isset.start_token_index = true;
        } else {
          xfer += iprot->skip(ftype);
        }
        break;
      case 4:
        if (ftype == ::apache::thrift::protocol::T_I32) {
          xfer += iprot->readI32(this->end_token_index);
          this->__isset.end_token_index = true;
        } else {
          xfer += iprot->skip(ftype);
        }
        break;
      case 5:
        if (ftype == ::apache::thrift::protocol::T_I32) {
          xfer += iprot->readI32(this->wordvec_dim);
          this->__isset.wordvec_dim = true;
        } else {
          xfer += iprot->skip(ftype);
        }
        break;
      case 6:
        if (ftype == ::apache::thrift::protocol::T_I32) {
          xfer += iprot->readI32(this->hidden_dim);
          this->__isset.hidden_dim = true;
        } else {
          xfer += iprot->skip(ftype);
        }
        break;
      default:
        xfer += iprot->skip(ftype);
        break;
    }
    xfer += iprot->readFieldEnd();
  }

  xfer += iprot->readStructEnd();

  return xfer;
}

uint32_t ModelInfo::write(::apache::thrift::protocol::TProtocol* oprot) const {
  uint32_t xfer = 0;
  xfer += oprot->writeStructBegin("ModelInfo");

  xfer += oprot->writeFieldBegin("window_size", ::apache::thrift::protocol::T_I32, 1);
  xfer += oprot->writeI32(this->window_size);
  xfer += oprot->writeFieldEnd();

  xfer += oprot->writeFieldBegin("vocab_size", ::apache::thrift::protocol::T_I32, 2);
  xfer += oprot->writeI32(this->vocab_size);
  xfer += oprot->writeFieldEnd();

  xfer += oprot->writeFieldBegin("start_token_index", ::apache::thrift::protocol::T_I32, 3);
  xfer += oprot->writeI32(this->start_token_index);
  xfer += oprot->writeFieldEnd();

  xfer += oprot->writeFieldBegin("end_token_index", ::apache::thrift::protocol::T_I32, 4);
  xfer += oprot->writeI32(this->end_token_index);
  xfer += oprot->writeFieldEnd();

  xfer += oprot->writeFieldBegin("wordvec_dim", ::apache::thrift::protocol::T_I32, 5);
  xfer += oprot->writeI32(this->wordvec_dim);
  xfer += oprot->writeFieldEnd();

  xfer += oprot->writeFieldBegin("hidden_dim", ::apache::thrift::protocol::T_I32, 6);
  xfer += oprot->writeI32(this->hidden_dim);
  xfer += oprot->writeFieldEnd();

  xfer += oprot->writeFieldStop();
  xfer += oprot->writeStructEnd();
  return xfer;
}

void swap(ModelInfo &a, ModelInfo &b) {
  using ::std::swap;
  swap(a.window_size, b.window_size);
  swap(a.vocab_size, b.vocab_size);
  swap(a.start_token_index, b.start_token_index);
  swap(a.end_token_index, b.end_token_index);
  swap(a.wordvec_dim, b.wordvec_dim);
  swap(a.hidden_dim, b.hidden_dim);
  swap(a.__isset, b.__isset);
}

const char* Params::ascii_fingerprint = "882E5A0F5FB0D66D6CADC597860F19AF";
const uint8_t Params::binary_fingerprint[16] = {0x88,0x2E,0x5A,0x0F,0x5F,0xB0,0xD6,0x6D,0x6C,0xAD,0xC5,0x97,0x86,0x0F,0x19,0xAF};

uint32_t Params::read(::apache::thrift::protocol::TProtocol* iprot) {

  uint32_t xfer = 0;
  std::string fname;
  ::apache::thrift::protocol::TType ftype;
  int16_t fid;

  xfer += iprot->readStructBegin(fname);

  using ::apache::thrift::protocol::TProtocolException;


  while (true)
  {
    xfer += iprot->readFieldBegin(fname, ftype, fid);
    if (ftype == ::apache::thrift::protocol::T_STOP) {
      break;
    }
    switch (fid)
    {
      case 1:
        if (ftype == ::apache::thrift::protocol::T_LIST) {
          {
            this->wordvec_weights.clear();
            uint32_t _size0;
            ::apache::thrift::protocol::TType _etype3;
            xfer += iprot->readListBegin(_etype3, _size0);
            this->wordvec_weights.resize(_size0);
            uint32_t _i4;
            for (_i4 = 0; _i4 < _size0; ++_i4)
            {
              xfer += iprot->readDouble(this->wordvec_weights[_i4]);
            }
            xfer += iprot->readListEnd();
          }
          this->__isset.wordvec_weights = true;
        } else {
          xfer += iprot->skip(ftype);
        }
        break;
      case 2:
        if (ftype == ::apache::thrift::protocol::T_LIST) {
          {
            this->input_hidden_weights.clear();
            uint32_t _size5;
            ::apache::thrift::protocol::TType _etype8;
            xfer += iprot->readListBegin(_etype8, _size5);
            this->input_hidden_weights.resize(_size5);
            uint32_t _i9;
            for (_i9 = 0; _i9 < _size5; ++_i9)
            {
              xfer += iprot->readDouble(this->input_hidden_weights[_i9]);
            }
            xfer += iprot->readListEnd();
          }
          this->__isset.input_hidden_weights = true;
        } else {
          xfer += iprot->skip(ftype);
        }
        break;
      case 3:
        if (ftype == ::apache::thrift::protocol::T_LIST) {
          {
            this->input_hidden_biases.clear();
            uint32_t _size10;
            ::apache::thrift::protocol::TType _etype13;
            xfer += iprot->readListBegin(_etype13, _size10);
            this->input_hidden_biases.resize(_size10);
            uint32_t _i14;
            for (_i14 = 0; _i14 < _size10; ++_i14)
            {
              xfer += iprot->readDouble(this->input_hidden_biases[_i14]);
            }
            xfer += iprot->readListEnd();
          }
          this->__isset.input_hidden_biases = true;
        } else {
          xfer += iprot->skip(ftype);
        }
        break;
      case 4:
        if (ftype == ::apache::thrift::protocol::T_LIST) {
          {
            this->hidden_output_weights.clear();
            uint32_t _size15;
            ::apache::thrift::protocol::TType _etype18;
            xfer += iprot->readListBegin(_etype18, _size15);
            this->hidden_output_weights.resize(_size15);
            uint32_t _i19;
            for (_i19 = 0; _i19 < _size15; ++_i19)
            {
              xfer += iprot->readDouble(this->hidden_output_weights[_i19]);
            }
            xfer += iprot->readListEnd();
          }
          this->__isset.hidden_output_weights = true;
        } else {
          xfer += iprot->skip(ftype);
        }
        break;
      case 5:
        if (ftype == ::apache::thrift::protocol::T_LIST) {
          {
            this->hidden_output_biases.clear();
            uint32_t _size20;
            ::apache::thrift::protocol::TType _etype23;
            xfer += iprot->readListBegin(_etype23, _size20);
            this->hidden_output_biases.resize(_size20);
            uint32_t _i24;
            for (_i24 = 0; _i24 < _size20; ++_i24)
            {
              xfer += iprot->readDouble(this->hidden_output_biases[_i24]);
            }
            xfer += iprot->readListEnd();
          }
          this->__isset.hidden_output_biases = true;
        } else {
          xfer += iprot->skip(ftype);
        }
        break;
      default:
        xfer += iprot->skip(ftype);
        break;
    }
    xfer += iprot->readFieldEnd();
  }

  xfer += iprot->readStructEnd();

  return xfer;
}

uint32_t Params::write(::apache::thrift::protocol::TProtocol* oprot) const {
  uint32_t xfer = 0;
  xfer += oprot->writeStructBegin("Params");

  xfer += oprot->writeFieldBegin("wordvec_weights", ::apache::thrift::protocol::T_LIST, 1);
  {
    xfer += oprot->writeListBegin(::apache::thrift::protocol::T_DOUBLE, static_cast<uint32_t>(this->wordvec_weights.size()));
    std::vector<double> ::const_iterator _iter25;
    for (_iter25 = this->wordvec_weights.begin(); _iter25 != this->wordvec_weights.end(); ++_iter25)
    {
      xfer += oprot->writeDouble((*_iter25));
    }
    xfer += oprot->writeListEnd();
  }
  xfer += oprot->writeFieldEnd();

  xfer += oprot->writeFieldBegin("input_hidden_weights", ::apache::thrift::protocol::T_LIST, 2);
  {
    xfer += oprot->writeListBegin(::apache::thrift::protocol::T_DOUBLE, static_cast<uint32_t>(this->input_hidden_weights.size()));
    std::vector<double> ::const_iterator _iter26;
    for (_iter26 = this->input_hidden_weights.begin(); _iter26 != this->input_hidden_weights.end(); ++_iter26)
    {
      xfer += oprot->writeDouble((*_iter26));
    }
    xfer += oprot->writeListEnd();
  }
  xfer += oprot->writeFieldEnd();

  xfer += oprot->writeFieldBegin("input_hidden_biases", ::apache::thrift::protocol::T_LIST, 3);
  {
    xfer += oprot->writeListBegin(::apache::thrift::protocol::T_DOUBLE, static_cast<uint32_t>(this->input_hidden_biases.size()));
    std::vector<double> ::const_iterator _iter27;
    for (_iter27 = this->input_hidden_biases.begin(); _iter27 != this->input_hidden_biases.end(); ++_iter27)
    {
      xfer += oprot->writeDouble((*_iter27));
    }
    xfer += oprot->writeListEnd();
  }
  xfer += oprot->writeFieldEnd();

  xfer += oprot->writeFieldBegin("hidden_output_weights", ::apache::thrift::protocol::T_LIST, 4);
  {
    xfer += oprot->writeListBegin(::apache::thrift::protocol::T_DOUBLE, static_cast<uint32_t>(this->hidden_output_weights.size()));
    std::vector<double> ::const_iterator _iter28;
    for (_iter28 = this->hidden_output_weights.begin(); _iter28 != this->hidden_output_weights.end(); ++_iter28)
    {
      xfer += oprot->writeDouble((*_iter28));
    }
    xfer += oprot->writeListEnd();
  }
  xfer += oprot->writeFieldEnd();

  xfer += oprot->writeFieldBegin("hidden_output_biases", ::apache::thrift::protocol::T_LIST, 5);
  {
    xfer += oprot->writeListBegin(::apache::thrift::protocol::T_DOUBLE, static_cast<uint32_t>(this->hidden_output_biases.size()));
    std::vector<double> ::const_iterator _iter29;
    for (_iter29 = this->hidden_output_biases.begin(); _iter29 != this->hidden_output_biases.end(); ++_iter29)
    {
      xfer += oprot->writeDouble((*_iter29));
    }
    xfer += oprot->writeListEnd();
  }
  xfer += oprot->writeFieldEnd();

  xfer += oprot->writeFieldStop();
  xfer += oprot->writeStructEnd();
  return xfer;
}

void swap(Params &a, Params &b) {
  using ::std::swap;
  swap(a.wordvec_weights, b.wordvec_weights);
  swap(a.input_hidden_weights, b.input_hidden_weights);
  swap(a.input_hidden_biases, b.input_hidden_biases);
  swap(a.hidden_output_weights, b.hidden_output_weights);
  swap(a.hidden_output_biases, b.hidden_output_biases);
  swap(a.__isset, b.__isset);
}

const char* ServerInfo::ascii_fingerprint = "EEBC915CE44901401D881E6091423036";
const uint8_t ServerInfo::binary_fingerprint[16] = {0xEE,0xBC,0x91,0x5C,0xE4,0x49,0x01,0x40,0x1D,0x88,0x1E,0x60,0x91,0x42,0x30,0x36};

uint32_t ServerInfo::read(::apache::thrift::protocol::TProtocol* iprot) {

  uint32_t xfer = 0;
  std::string fname;
  ::apache::thrift::protocol::TType ftype;
  int16_t fid;

  xfer += iprot->readStructBegin(fname);

  using ::apache::thrift::protocol::TProtocolException;


  while (true)
  {
    xfer += iprot->readFieldBegin(fname, ftype, fid);
    if (ftype == ::apache::thrift::protocol::T_STOP) {
      break;
    }
    switch (fid)
    {
      case 1:
        if (ftype == ::apache::thrift::protocol::T_STRING) {
          xfer += iprot->readString(this->ip);
          this->__isset.ip = true;
        } else {
          xfer += iprot->skip(ftype);
        }
        break;
      case 2:
        if (ftype == ::apache::thrift::protocol::T_I32) {
          xfer += iprot->readI32(this->port);
          this->__isset.port = true;
        } else {
          xfer += iprot->skip(ftype);
        }
        break;
      default:
        xfer += iprot->skip(ftype);
        break;
    }
    xfer += iprot->readFieldEnd();
  }

  xfer += iprot->readStructEnd();

  return xfer;
}

uint32_t ServerInfo::write(::apache::thrift::protocol::TProtocol* oprot) const {
  uint32_t xfer = 0;
  xfer += oprot->writeStructBegin("ServerInfo");

  xfer += oprot->writeFieldBegin("ip", ::apache::thrift::protocol::T_STRING, 1);
  xfer += oprot->writeString(this->ip);
  xfer += oprot->writeFieldEnd();

  xfer += oprot->writeFieldBegin("port", ::apache::thrift::protocol::T_I32, 2);
  xfer += oprot->writeI32(this->port);
  xfer += oprot->writeFieldEnd();

  xfer += oprot->writeFieldStop();
  xfer += oprot->writeStructEnd();
  return xfer;
}

void swap(ServerInfo &a, ServerInfo &b) {
  using ::std::swap;
  swap(a.ip, b.ip);
  swap(a.port, b.port);
  swap(a.__isset, b.__isset);
}

const char* AnnounceRequest::ascii_fingerprint = "2BD9E1CC52BCB0899198EEADB3593B00";
const uint8_t AnnounceRequest::binary_fingerprint[16] = {0x2B,0xD9,0xE1,0xCC,0x52,0xBC,0xB0,0x89,0x91,0x98,0xEE,0xAD,0xB3,0x59,0x3B,0x00};

uint32_t AnnounceRequest::read(::apache::thrift::protocol::TProtocol* iprot) {

  uint32_t xfer = 0;
  std::string fname;
  ::apache::thrift::protocol::TType ftype;
  int16_t fid;

  xfer += iprot->readStructBegin(fname);

  using ::apache::thrift::protocol::TProtocolException;


  while (true)
  {
    xfer += iprot->readFieldBegin(fname, ftype, fid);
    if (ftype == ::apache::thrift::protocol::T_STOP) {
      break;
    }
    switch (fid)
    {
      case 1:
        if (ftype == ::apache::thrift::protocol::T_STRUCT) {
          xfer += this->worker_info.read(iprot);
          this->__isset.worker_info = true;
        } else {
          xfer += iprot->skip(ftype);
        }
        break;
      default:
        xfer += iprot->skip(ftype);
        break;
    }
    xfer += iprot->readFieldEnd();
  }

  xfer += iprot->readStructEnd();

  return xfer;
}

uint32_t AnnounceRequest::write(::apache::thrift::protocol::TProtocol* oprot) const {
  uint32_t xfer = 0;
  xfer += oprot->writeStructBegin("AnnounceRequest");

  xfer += oprot->writeFieldBegin("worker_info", ::apache::thrift::protocol::T_STRUCT, 1);
  xfer += this->worker_info.write(oprot);
  xfer += oprot->writeFieldEnd();

  xfer += oprot->writeFieldStop();
  xfer += oprot->writeStructEnd();
  return xfer;
}

void swap(AnnounceRequest &a, AnnounceRequest &b) {
  using ::std::swap;
  swap(a.worker_info, b.worker_info);
  swap(a.__isset, b.__isset);
}

const char* AnnounceResponse::ascii_fingerprint = "AFBEB144070F2944B0ABB54DD68305D0";
const uint8_t AnnounceResponse::binary_fingerprint[16] = {0xAF,0xBE,0xB1,0x44,0x07,0x0F,0x29,0x44,0xB0,0xAB,0xB5,0x4D,0xD6,0x83,0x05,0xD0};

uint32_t AnnounceResponse::read(::apache::thrift::protocol::TProtocol* iprot) {

  uint32_t xfer = 0;
  std::string fname;
  ::apache::thrift::protocol::TType ftype;
  int16_t fid;

  xfer += iprot->readStructBegin(fname);

  using ::apache::thrift::protocol::TProtocolException;


  while (true)
  {
    xfer += iprot->readFieldBegin(fname, ftype, fid);
    if (ftype == ::apache::thrift::protocol::T_STOP) {
      break;
    }
    switch (fid)
    {
      case 1:
        if (ftype == ::apache::thrift::protocol::T_STRUCT) {
          xfer += this->model_info.read(iprot);
          this->__isset.model_info = true;
        } else {
          xfer += iprot->skip(ftype);
        }
        break;
      case 2:
        if (ftype == ::apache::thrift::protocol::T_STRUCT) {
          xfer += this->params.read(iprot);
          this->__isset.params = true;
        } else {
          xfer += iprot->skip(ftype);
        }
        break;
      case 3:
        if (ftype == ::apache::thrift::protocol::T_LIST) {
          {
            this->shard_paths.clear();
            uint32_t _size30;
            ::apache::thrift::protocol::TType _etype33;
            xfer += iprot->readListBegin(_etype33, _size30);
            this->shard_paths.resize(_size30);
            uint32_t _i34;
            for (_i34 = 0; _i34 < _size30; ++_i34)
            {
              xfer += iprot->readString(this->shard_paths[_i34]);
            }
            xfer += iprot->readListEnd();
          }
          this->__isset.shard_paths = true;
        } else {
          xfer += iprot->skip(ftype);
        }
        break;
      case 4:
        if (ftype == ::apache::thrift::protocol::T_DOUBLE) {
          xfer += iprot->readDouble(this->learn_rate);
          this->__isset.learn_rate = true;
        } else {
          xfer += iprot->skip(ftype);
        }
        break;
      case 5:
        if (ftype == ::apache::thrift::protocol::T_LIST) {
          {
            this->param_servers.clear();
            uint32_t _size35;
            ::apache::thrift::protocol::TType _etype38;
            xfer += iprot->readListBegin(_etype38, _size35);
            this->param_servers.resize(_size35);
            uint32_t _i39;
            for (_i39 = 0; _i39 < _size35; ++_i39)
            {
              xfer += this->param_servers[_i39].read(iprot);
            }
            xfer += iprot->readListEnd();
          }
          this->__isset.param_servers = true;
        } else {
          xfer += iprot->skip(ftype);
        }
        break;
      default:
        xfer += iprot->skip(ftype);
        break;
    }
    xfer += iprot->readFieldEnd();
  }

  xfer += iprot->readStructEnd();

  return xfer;
}

uint32_t AnnounceResponse::write(::apache::thrift::protocol::TProtocol* oprot) const {
  uint32_t xfer = 0;
  xfer += oprot->writeStructBegin("AnnounceResponse");

  xfer += oprot->writeFieldBegin("model_info", ::apache::thrift::protocol::T_STRUCT, 1);
  xfer += this->model_info.write(oprot);
  xfer += oprot->writeFieldEnd();

  xfer += oprot->writeFieldBegin("params", ::apache::thrift::protocol::T_STRUCT, 2);
  xfer += this->params.write(oprot);
  xfer += oprot->writeFieldEnd();

  xfer += oprot->writeFieldBegin("shard_paths", ::apache::thrift::protocol::T_LIST, 3);
  {
    xfer += oprot->writeListBegin(::apache::thrift::protocol::T_STRING, static_cast<uint32_t>(this->shard_paths.size()));
    std::vector<std::string> ::const_iterator _iter40;
    for (_iter40 = this->shard_paths.begin(); _iter40 != this->shard_paths.end(); ++_iter40)
    {
      xfer += oprot->writeString((*_iter40));
    }
    xfer += oprot->writeListEnd();
  }
  xfer += oprot->writeFieldEnd();

  xfer += oprot->writeFieldBegin("learn_rate", ::apache::thrift::protocol::T_DOUBLE, 4);
  xfer += oprot->writeDouble(this->learn_rate);
  xfer += oprot->writeFieldEnd();

  xfer += oprot->writeFieldBegin("param_servers", ::apache::thrift::protocol::T_LIST, 5);
  {
    xfer += oprot->writeListBegin(::apache::thrift::protocol::T_STRUCT, static_cast<uint32_t>(this->param_servers.size()));
    std::vector<ServerInfo> ::const_iterator _iter41;
    for (_iter41 = this->param_servers.begin(); _iter41 != this->param_servers.end(); ++_iter41)
    {
      xfer += (*_iter41).write(oprot);
    }
    xfer += oprot->writeListEnd();
  }
  xfer += oprot->writeFieldEnd();

  xfer += oprot->writeFieldStop();
  xfer += oprot->writeStructEnd();
  return xfer;
}

void swap(AnnounceResponse &a, AnnounceResponse &b) {
  using ::std::swap;
  swap(a.model_info, b.model_info);
  swap(a.params, b.params);
  swap(a.shard_paths, b.shard_paths);
  swap(a.learn_rate, b.learn_rate);
  swap(a.param_servers, b.param_servers);
  swap(a.__isset, b.__isset);
}

const char* UpdateRequest::ascii_fingerprint = "5252BFD9FD3B10A47FE26840939EECE6";
const uint8_t UpdateRequest::binary_fingerprint[16] = {0x52,0x52,0xBF,0xD9,0xFD,0x3B,0x10,0xA4,0x7F,0xE2,0x68,0x40,0x93,0x9E,0xEC,0xE6};

uint32_t UpdateRequest::read(::apache::thrift::protocol::TProtocol* iprot) {

  uint32_t xfer = 0;
  std::string fname;
  ::apache::thrift::protocol::TType ftype;
  int16_t fid;

  xfer += iprot->readStructBegin(fname);

  using ::apache::thrift::protocol::TProtocolException;


  while (true)
  {
    xfer += iprot->readFieldBegin(fname, ftype, fid);
    if (ftype == ::apache::thrift::protocol::T_STOP) {
      break;
    }
    switch (fid)
    {
      case 1:
        if (ftype == ::apache::thrift::protocol::T_STRUCT) {
          xfer += this->update.read(iprot);
          this->__isset.update = true;
        } else {
          xfer += iprot->skip(ftype);
        }
        break;
      case 2:
        if (ftype == ::apache::thrift::protocol::T_STRUCT) {
          xfer += this->worker_info.read(iprot);
          this->__isset.worker_info = true;
        } else {
          xfer += iprot->skip(ftype);
        }
        break;
      default:
        xfer += iprot->skip(ftype);
        break;
    }
    xfer += iprot->readFieldEnd();
  }

  xfer += iprot->readStructEnd();

  return xfer;
}

uint32_t UpdateRequest::write(::apache::thrift::protocol::TProtocol* oprot) const {
  uint32_t xfer = 0;
  xfer += oprot->writeStructBegin("UpdateRequest");

  xfer += oprot->writeFieldBegin("update", ::apache::thrift::protocol::T_STRUCT, 1);
  xfer += this->update.write(oprot);
  xfer += oprot->writeFieldEnd();

  if (this->__isset.worker_info) {
    xfer += oprot->writeFieldBegin("worker_info", ::apache::thrift::protocol::T_STRUCT, 2);
    xfer += this->worker_info.write(oprot);
    xfer += oprot->writeFieldEnd();
  }
  xfer += oprot->writeFieldStop();
  xfer += oprot->writeStructEnd();
  return xfer;
}

void swap(UpdateRequest &a, UpdateRequest &b) {
  using ::std::swap;
  swap(a.update, b.update);
  swap(a.worker_info, b.worker_info);
  swap(a.__isset, b.__isset);
}

const char* UpdateResponse::ascii_fingerprint = "E3BAB2DD3B420D9247DE4278A71DC1B0";
const uint8_t UpdateResponse::binary_fingerprint[16] = {0xE3,0xBA,0xB2,0xDD,0x3B,0x42,0x0D,0x92,0x47,0xDE,0x42,0x78,0xA7,0x1D,0xC1,0xB0};

uint32_t UpdateResponse::read(::apache::thrift::protocol::TProtocol* iprot) {

  uint32_t xfer = 0;
  std::string fname;
  ::apache::thrift::protocol::TType ftype;
  int16_t fid;

  xfer += iprot->readStructBegin(fname);

  using ::apache::thrift::protocol::TProtocolException;


  while (true)
  {
    xfer += iprot->readFieldBegin(fname, ftype, fid);
    if (ftype == ::apache::thrift::protocol::T_STOP) {
      break;
    }
    switch (fid)
    {
      case 1:
        if (ftype == ::apache::thrift::protocol::T_LIST) {
          {
            this->param_servers.clear();
            uint32_t _size42;
            ::apache::thrift::protocol::TType _etype45;
            xfer += iprot->readListBegin(_etype45, _size42);
            this->param_servers.resize(_size42);
            uint32_t _i46;
            for (_i46 = 0; _i46 < _size42; ++_i46)
            {
              xfer += this->param_servers[_i46].read(iprot);
            }
            xfer += iprot->readListEnd();
          }
          this->__isset.param_servers = true;
        } else {
          xfer += iprot->skip(ftype);
        }
        break;
      default:
        xfer += iprot->skip(ftype);
        break;
    }
    xfer += iprot->readFieldEnd();
  }

  xfer += iprot->readStructEnd();

  return xfer;
}

uint32_t UpdateResponse::write(::apache::thrift::protocol::TProtocol* oprot) const {
  uint32_t xfer = 0;
  xfer += oprot->writeStructBegin("UpdateResponse");

  if (this->__isset.param_servers) {
    xfer += oprot->writeFieldBegin("param_servers", ::apache::thrift::protocol::T_LIST, 1);
    {
      xfer += oprot->writeListBegin(::apache::thrift::protocol::T_STRUCT, static_cast<uint32_t>(this->param_servers.size()));
      std::vector<ServerInfo> ::const_iterator _iter47;
      for (_iter47 = this->param_servers.begin(); _iter47 != this->param_servers.end(); ++_iter47)
      {
        xfer += (*_iter47).write(oprot);
      }
      xfer += oprot->writeListEnd();
    }
    xfer += oprot->writeFieldEnd();
  }
  xfer += oprot->writeFieldStop();
  xfer += oprot->writeStructEnd();
  return xfer;
}

void swap(UpdateResponse &a, UpdateResponse &b) {
  using ::std::swap;
  swap(a.param_servers, b.param_servers);
  swap(a.__isset, b.__isset);
}

const char* PullRequest::ascii_fingerprint = "45F07C5FAF8C5003DA143A45C267ADFE";
const uint8_t PullRequest::binary_fingerprint[16] = {0x45,0xF0,0x7C,0x5F,0xAF,0x8C,0x50,0x03,0xDA,0x14,0x3A,0x45,0xC2,0x67,0xAD,0xFE};

uint32_t PullRequest::read(::apache::thrift::protocol::TProtocol* iprot) {

  uint32_t xfer = 0;
  std::string fname;
  ::apache::thrift::protocol::TType ftype;
  int16_t fid;

  xfer += iprot->readStructBegin(fname);

  using ::apache::thrift::protocol::TProtocolException;


  while (true)
  {
    xfer += iprot->readFieldBegin(fname, ftype, fid);
    if (ftype == ::apache::thrift::protocol::T_STOP) {
      break;
    }
    switch (fid)
    {
      case 1:
        if (ftype == ::apache::thrift::protocol::T_STRUCT) {
          xfer += this->worker_info.read(iprot);
          this->__isset.worker_info = true;
        } else {
          xfer += iprot->skip(ftype);
        }
        break;
      default:
        xfer += iprot->skip(ftype);
        break;
    }
    xfer += iprot->readFieldEnd();
  }

  xfer += iprot->readStructEnd();

  return xfer;
}

uint32_t PullRequest::write(::apache::thrift::protocol::TProtocol* oprot) const {
  uint32_t xfer = 0;
  xfer += oprot->writeStructBegin("PullRequest");

  if (this->__isset.worker_info) {
    xfer += oprot->writeFieldBegin("worker_info", ::apache::thrift::protocol::T_STRUCT, 1);
    xfer += this->worker_info.write(oprot);
    xfer += oprot->writeFieldEnd();
  }
  xfer += oprot->writeFieldStop();
  xfer += oprot->writeStructEnd();
  return xfer;
}

void swap(PullRequest &a, PullRequest &b) {
  using ::std::swap;
  swap(a.worker_info, b.worker_info);
  swap(a.__isset, b.__isset);
}

const char* PullResponse::ascii_fingerprint = "76FE0E92C52E0EC69A75E4CF3AE83DEF";
const uint8_t PullResponse::binary_fingerprint[16] = {0x76,0xFE,0x0E,0x92,0xC5,0x2E,0x0E,0xC6,0x9A,0x75,0xE4,0xCF,0x3A,0xE8,0x3D,0xEF};

uint32_t PullResponse::read(::apache::thrift::protocol::TProtocol* iprot) {

  uint32_t xfer = 0;
  std::string fname;
  ::apache::thrift::protocol::TType ftype;
  int16_t fid;

  xfer += iprot->readStructBegin(fname);

  using ::apache::thrift::protocol::TProtocolException;


  while (true)
  {
    xfer += iprot->readFieldBegin(fname, ftype, fid);
    if (ftype == ::apache::thrift::protocol::T_STOP) {
      break;
    }
    switch (fid)
    {
      case 1:
        if (ftype == ::apache::thrift::protocol::T_STRUCT) {
          xfer += this->params.read(iprot);
          this->__isset.params = true;
        } else {
          xfer += iprot->skip(ftype);
        }
        break;
      case 2:
        if (ftype == ::apache::thrift::protocol::T_LIST) {
          {
            this->param_servers.clear();
            uint32_t _size48;
            ::apache::thrift::protocol::TType _etype51;
            xfer += iprot->readListBegin(_etype51, _size48);
            this->param_servers.resize(_size48);
            uint32_t _i52;
            for (_i52 = 0; _i52 < _size48; ++_i52)
            {
              xfer += this->param_servers[_i52].read(iprot);
            }
            xfer += iprot->readListEnd();
          }
          this->__isset.param_servers = true;
        } else {
          xfer += iprot->skip(ftype);
        }
        break;
      default:
        xfer += iprot->skip(ftype);
        break;
    }
    xfer += iprot->readFieldEnd();
  }

  xfer += iprot->readStructEnd();

  return xfer;
}

uint32_t PullResponse::write(::apache::thrift::protocol::TProtocol* oprot) const {
  uint32_t xfer = 0;
  xfer += oprot->writeStructBegin("PullResponse");

  xfer += oprot->writeFieldBegin("params", ::apache::thrift::protocol::T_STRUCT, 1);
  xfer += this->params.write(oprot);
  xfer += oprot->writeFieldEnd();

  if (this->__isset.param_servers) {
    xfer += oprot->writeFieldBegin("param_servers", ::apache::thrift::protocol::T_LIST, 2);
    {
      xfer += oprot->writeListBegin(::apache::thrift::protocol::T_STRUCT, static_cast<uint32_t>(this->param_servers.size()));
      std::vector<ServerInfo> ::const_iterator _iter53;
      for (_iter53 = this->param_servers.begin(); _iter53 != this->param_servers.end(); ++_iter53)
      {
        xfer += (*_iter53).write(oprot);
      }
      xfer += oprot->writeListEnd();
    }
    xfer += oprot->writeFieldEnd();
  }
  xfer += oprot->writeFieldStop();
  xfer += oprot->writeStructEnd();
  return xfer;
}

void swap(PullResponse &a, PullResponse &b) {
  using ::std::swap;
  swap(a.params, b.params);
  swap(a.param_servers, b.param_servers);
  swap(a.__isset, b.__isset);
}

const char* HBRequest::ascii_fingerprint = "45F07C5FAF8C5003DA143A45C267ADFE";
const uint8_t HBRequest::binary_fingerprint[16] = {0x45,0xF0,0x7C,0x5F,0xAF,0x8C,0x50,0x03,0xDA,0x14,0x3A,0x45,0xC2,0x67,0xAD,0xFE};

uint32_t HBRequest::read(::apache::thrift::protocol::TProtocol* iprot) {

  uint32_t xfer = 0;
  std::string fname;
  ::apache::thrift::protocol::TType ftype;
  int16_t fid;

  xfer += iprot->readStructBegin(fname);

  using ::apache::thrift::protocol::TProtocolException;


  while (true)
  {
    xfer += iprot->readFieldBegin(fname, ftype, fid);
    if (ftype == ::apache::thrift::protocol::T_STOP) {
      break;
    }
    switch (fid)
    {
      case 1:
        if (ftype == ::apache::thrift::protocol::T_STRUCT) {
          xfer += this->master_info.read(iprot);
          this->__isset.master_info = true;
        } else {
          xfer += iprot->skip(ftype);
        }
        break;
      default:
        xfer += iprot->skip(ftype);
        break;
    }
    xfer += iprot->readFieldEnd();
  }

  xfer += iprot->readStructEnd();

  return xfer;
}

uint32_t HBRequest::write(::apache::thrift::protocol::TProtocol* oprot) const {
  uint32_t xfer = 0;
  xfer += oprot->writeStructBegin("HBRequest");

  if (this->__isset.master_info) {
    xfer += oprot->writeFieldBegin("master_info", ::apache::thrift::protocol::T_STRUCT, 1);
    xfer += this->master_info.write(oprot);
    xfer += oprot->writeFieldEnd();
  }
  xfer += oprot->writeFieldStop();
  xfer += oprot->writeStructEnd();
  return xfer;
}

void swap(HBRequest &a, HBRequest &b) {
  using ::std::swap;
  swap(a.master_info, b.master_info);
  swap(a.__isset, b.__isset);
}

const char* HBResponse::ascii_fingerprint = "45F07C5FAF8C5003DA143A45C267ADFE";
const uint8_t HBResponse::binary_fingerprint[16] = {0x45,0xF0,0x7C,0x5F,0xAF,0x8C,0x50,0x03,0xDA,0x14,0x3A,0x45,0xC2,0x67,0xAD,0xFE};

uint32_t HBResponse::read(::apache::thrift::protocol::TProtocol* iprot) {

  uint32_t xfer = 0;
  std::string fname;
  ::apache::thrift::protocol::TType ftype;
  int16_t fid;

  xfer += iprot->readStructBegin(fname);

  using ::apache::thrift::protocol::TProtocolException;


  while (true)
  {
    xfer += iprot->readFieldBegin(fname, ftype, fid);
    if (ftype == ::apache::thrift::protocol::T_STOP) {
      break;
    }
    switch (fid)
    {
      case 1:
        if (ftype == ::apache::thrift::protocol::T_STRUCT) {
          xfer += this->worker_info.read(iprot);
          this->__isset.worker_info = true;
        } else {
          xfer += iprot->skip(ftype);
        }
        break;
      default:
        xfer += iprot->skip(ftype);
        break;
    }
    xfer += iprot->readFieldEnd();
  }

  xfer += iprot->readStructEnd();

  return xfer;
}

uint32_t HBResponse::write(::apache::thrift::protocol::TProtocol* oprot) const {
  uint32_t xfer = 0;
  xfer += oprot->writeStructBegin("HBResponse");

  if (this->__isset.worker_info) {
    xfer += oprot->writeFieldBegin("worker_info", ::apache::thrift::protocol::T_STRUCT, 1);
    xfer += this->worker_info.write(oprot);
    xfer += oprot->writeFieldEnd();
  }
  xfer += oprot->writeFieldStop();
  xfer += oprot->writeStructEnd();
  return xfer;
}

void swap(HBResponse &a, HBResponse &b) {
  using ::std::swap;
  swap(a.worker_info, b.worker_info);
  swap(a.__isset, b.__isset);
}

const char* StartRequest::ascii_fingerprint = "F96E9B21D37ECC1E6B1D9CC351B177C9";
const uint8_t StartRequest::binary_fingerprint[16] = {0xF9,0x6E,0x9B,0x21,0xD3,0x7E,0xCC,0x1E,0x6B,0x1D,0x9C,0xC3,0x51,0xB1,0x77,0xC9};

uint32_t StartRequest::read(::apache::thrift::protocol::TProtocol* iprot) {

  uint32_t xfer = 0;
  std::string fname;
  ::apache::thrift::protocol::TType ftype;
  int16_t fid;

  xfer += iprot->readStructBegin(fname);

  using ::apache::thrift::protocol::TProtocolException;


  while (true)
  {
    xfer += iprot->readFieldBegin(fname, ftype, fid);
    if (ftype == ::apache::thrift::protocol::T_STOP) {
      break;
    }
    switch (fid)
    {
      case 1:
        if (ftype == ::apache::thrift::protocol::T_LIST) {
          {
            this->shard_paths.clear();
            uint32_t _size54;
            ::apache::thrift::protocol::TType _etype57;
            xfer += iprot->readListBegin(_etype57, _size54);
            this->shard_paths.resize(_size54);
            uint32_t _i58;
            for (_i58 = 0; _i58 < _size54; ++_i58)
            {
              xfer += iprot->readString(this->shard_paths[_i58]);
            }
            xfer += iprot->readListEnd();
          }
          this->__isset.shard_paths = true;
        } else {
          xfer += iprot->skip(ftype);
        }
        break;
      case 2:
        if (ftype == ::apache::thrift::protocol::T_DOUBLE) {
          xfer += iprot->readDouble(this->learn_rate);
          this->__isset.learn_rate = true;
        } else {
          xfer += iprot->skip(ftype);
        }
        break;
      default:
        xfer += iprot->skip(ftype);
        break;
    }
    xfer += iprot->readFieldEnd();
  }

  xfer += iprot->readStructEnd();

  return xfer;
}

uint32_t StartRequest::write(::apache::thrift::protocol::TProtocol* oprot) const {
  uint32_t xfer = 0;
  xfer += oprot->writeStructBegin("StartRequest");

  xfer += oprot->writeFieldBegin("shard_paths", ::apache::thrift::protocol::T_LIST, 1);
  {
    xfer += oprot->writeListBegin(::apache::thrift::protocol::T_STRING, static_cast<uint32_t>(this->shard_paths.size()));
    std::vector<std::string> ::const_iterator _iter59;
    for (_iter59 = this->shard_paths.begin(); _iter59 != this->shard_paths.end(); ++_iter59)
    {
      xfer += oprot->writeString((*_iter59));
    }
    xfer += oprot->writeListEnd();
  }
  xfer += oprot->writeFieldEnd();

  xfer += oprot->writeFieldBegin("learn_rate", ::apache::thrift::protocol::T_DOUBLE, 2);
  xfer += oprot->writeDouble(this->learn_rate);
  xfer += oprot->writeFieldEnd();

  xfer += oprot->writeFieldStop();
  xfer += oprot->writeStructEnd();
  return xfer;
}

void swap(StartRequest &a, StartRequest &b) {
  using ::std::swap;
  swap(a.shard_paths, b.shard_paths);
  swap(a.learn_rate, b.learn_rate);
  swap(a.__isset, b.__isset);
}

const char* ReassignRequest::ascii_fingerprint = "ACE4F644F0FDD289DDC4EE5B83BC13C0";
const uint8_t ReassignRequest::binary_fingerprint[16] = {0xAC,0xE4,0xF6,0x44,0xF0,0xFD,0xD2,0x89,0xDD,0xC4,0xEE,0x5B,0x83,0xBC,0x13,0xC0};

uint32_t ReassignRequest::read(::apache::thrift::protocol::TProtocol* iprot) {

  uint32_t xfer = 0;
  std::string fname;
  ::apache::thrift::protocol::TType ftype;
  int16_t fid;

  xfer += iprot->readStructBegin(fname);

  using ::apache::thrift::protocol::TProtocolException;


  while (true)
  {
    xfer += iprot->readFieldBegin(fname, ftype, fid);
    if (ftype == ::apache::thrift::protocol::T_STOP) {
      break;
    }
    switch (fid)
    {
      case 1:
        if (ftype == ::apache::thrift::protocol::T_LIST) {
          {
            this->shard_paths.clear();
            uint32_t _size60;
            ::apache::thrift::protocol::TType _etype63;
            xfer += iprot->readListBegin(_etype63, _size60);
            this->shard_paths.resize(_size60);
            uint32_t _i64;
            for (_i64 = 0; _i64 < _size60; ++_i64)
            {
              xfer += iprot->readString(this->shard_paths[_i64]);
            }
            xfer += iprot->readListEnd();
          }
          this->__isset.shard_paths = true;
        } else {
          xfer += iprot->skip(ftype);
        }
        break;
      default:
        xfer += iprot->skip(ftype);
        break;
    }
    xfer += iprot->readFieldEnd();
  }

  xfer += iprot->readStructEnd();

  return xfer;
}

uint32_t ReassignRequest::write(::apache::thrift::protocol::TProtocol* oprot) const {
  uint32_t xfer = 0;
  xfer += oprot->writeStructBegin("ReassignRequest");

  xfer += oprot->writeFieldBegin("shard_paths", ::apache::thrift::protocol::T_LIST, 1);
  {
    xfer += oprot->writeListBegin(::apache::thrift::protocol::T_STRING, static_cast<uint32_t>(this->shard_paths.size()));
    std::vector<std::string> ::const_iterator _iter65;
    for (_iter65 = this->shard_paths.begin(); _iter65 != this->shard_paths.end(); ++_iter65)
    {
      xfer += oprot->writeString((*_iter65));
    }
    xfer += oprot->writeListEnd();
  }
  xfer += oprot->writeFieldEnd();

  xfer += oprot->writeFieldStop();
  xfer += oprot->writeStructEnd();
  return xfer;
}

void swap(ReassignRequest &a, ReassignRequest &b) {
  using ::std::swap;
  swap(a.shard_paths, b.shard_paths);
  swap(a.__isset, b.__isset);
}

} // namespace
