# Native Qwen Talker backbone build for the RTX 3090 container.
CUDA_PATH ?= $(if $(wildcard /usr/local/cuda-13.1),/usr/local/cuda-13.1,/usr/local/cuda)
NVCC ?= $(CUDA_PATH)/bin/nvcc
override ARCH = -gencode arch=compute_86,code=sm_86
NVCCFLAGS = -O3 $(ARCH) -lineinfo --compiler-options -fPIC -std=c++17 -I. -I$(CUDA_PATH)/include
CXXFLAGS = -O3 -fPIC -std=c++17 -I. -I$(CUDA_PATH)/include
LDFLAGS = -L$(CUDA_PATH)/lib64 -lcudart -lcublas
TORCH_CONFIG ?= torch_build_config.mk
-include $(TORCH_CONFIG)

LIB_TARGET = libqwen_talker.so
CUDA_OBJS = qwen_talker.o qwen_talker_kernels.o
ATEN_OBJS = qwen_talker_aten_attention.o qwen_talker_aten_rmsnorm.o
OBJS = $(CUDA_OBJS) $(ATEN_OBJS)

all: $(TORCH_CONFIG) $(LIB_TARGET)

$(TORCH_CONFIG): ../../tools/kernel_gates/export_torch_build_config.py
	python $< --output $@

qwen_talker.o: qwen_talker.cu qwen_talker.h qwen_talker_kernels.cuh qwen_talker_aten_attention.h qwen_talker_aten_rmsnorm.h generated/qwen_native_topology.h
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

qwen_talker_kernels.o: qwen_talker_kernels.cu qwen_talker_kernels.cuh generated/qwen_native_topology.h
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

qwen_talker_aten_attention.o: qwen_talker_aten_attention.cpp qwen_talker_aten_attention.h qwen_talker.h generated/qwen_native_topology.h $(TORCH_CONFIG)
	$(CXX) $(CXXFLAGS) $(TORCH_CXXFLAGS) -c $< -o $@

qwen_talker_aten_rmsnorm.o: qwen_talker_aten_rmsnorm.cpp qwen_talker_aten_rmsnorm.h qwen_talker.h generated/qwen_native_topology.h $(TORCH_CONFIG)
	$(CXX) $(CXXFLAGS) $(TORCH_CXXFLAGS) -c $< -o $@

$(LIB_TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) -shared $(OBJS) -o $@ $(LDFLAGS) $(TORCH_LDFLAGS)

clean:
	rm -f $(OBJS) $(LIB_TARGET) $(TORCH_CONFIG)

.PHONY: all clean
