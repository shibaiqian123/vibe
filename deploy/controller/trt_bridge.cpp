// C ABI for validation and deployment using SONIC's original TRTInferenceEngine.
#include "InferenceEngine.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <algorithm>
#include <cstring>
static thread_local std::string error;
struct Session { TRTInferenceEngine engine; cudaStream_t stream{};
 ~Session(){if(stream) cudaStreamDestroy(stream);} };
extern "C" {
const char* vibe_error(){return error.c_str();}
void* vibe_open(const char* path) {
 auto s=std::make_unique<Session>();
 try {Options o; o.precision=Precision::FP32; std::string cache;
  if(!ConvertONNXToTRT(o,path,cache,"vibe_fp32_",false)) throw std::runtime_error("SONIC ONNX->TRT conversion failed");
  if(!s->engine.Initialize(cache,0,{}) || !s->engine.InitInputs({})) throw std::runtime_error("TRT initialization failed");
  if(cudaStreamCreate(&s->stream)!=cudaSuccess) throw std::runtime_error("CUDA stream failed");
  return s.release();
 } catch(const std::exception& e){error=e.what();return nullptr;}
}
void vibe_close(void* p){delete static_cast<Session*>(p);}
// io=0 inputs, io=1 outputs. Names are newline-delimited.
const char* vibe_names(void* p,int io){static thread_local std::string names; names.clear();
 auto& e=static_cast<Session*>(p)->engine;
 for(auto& n:io?e.GetOutputTensorNames():e.GetInputTensorNames()) names+=n+"\n";
 return names.c_str();}
int vibe_shape(void* p,const char* name,int64_t* dims,int capacity){
 std::vector<int64_t> shape;
 if(!static_cast<Session*>(p)->engine.GetTensorShape(name,shape)||shape.size()>size_t(capacity)) return -1;
 std::copy(shape.begin(),shape.end(),dims);return int(shape.size());}
int vibe_type(void* p,const char* n){return int(static_cast<Session*>(p)->engine.GetTensorDataType(n));}
int vibe_transfer(void* p,const char* n,void* data,size_t bytes,int output){
 try{auto& e=static_cast<Session*>(p)->engine;
 if(output)e.GetOutputData(n,data,bytes);else e.SetInputData(n,data,bytes);return 0;
 }catch(const std::exception& e){error=e.what();return -1;}}
int vibe_run(void* p){try{auto* s=static_cast<Session*>(p);
 if(!s->engine.Enqueue(s->stream))throw std::runtime_error("Enqueue failed");
 auto result=cudaStreamSynchronize(s->stream);
 if(result!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(result));return 0;
 }catch(const std::exception& e){error=e.what();return -1;}}
}
