# Makefile — Builds libgemma4_listener.so and gemma4_listener_test
# Conforms to P4-NATIVE Implementation Plan §21

CUDA_PATH ?= $(if $(wildcard /usr/local/cuda-13.1),/usr/local/cuda-13.1,/usr/local/cuda)
NVCC ?= $(CUDA_PATH)/bin/nvcc

# This executor is certified only for the RTX 3090 target and deliberately
# does not enable --use_fast_math.
override ARCH = -gencode arch=compute_86,code=sm_86
NVCCFLAGS = -O3 $(ARCH) -lineinfo --compiler-options -fPIC -Xcompiler -fno-exceptions -std=c++17 -I. -I$(CUDA_PATH)/include
# CUTLASS_ORDERED_TILES_V1 transitively includes CUTLASS headers containing
# host-side diagnostic throws; this TU must retain normal exception support.
ORDERED_NVCCFLAGS = -O3 $(ARCH) -lineinfo --compiler-options -fPIC -std=c++17 -I. -I$(CUDA_PATH)/include
CXXFLAGS = -O3 -fPIC -std=c++17 -I. -I$(CUDA_PATH)/include
LDFLAGS = -L$(CUDA_PATH)/lib64 -lcudart -lcublas

# The ATen bridge is compiled against the exact PyTorch installed in the
# target container. Generate this file there; it is intentionally not guessed.
TORCH_CONFIG ?= torch_build_config.mk
-include $(TORCH_CONFIG)

LIB_TARGET = libgemma4_listener.so
TEST_TARGET = gemma4_listener_test

CUDA_SRCS = gemma4_listener.cu gemma4_listener_cache.cu gemma4_listener_rms_exact.cu
CUDA_OBJS = $(CUDA_SRCS:.cu=.o)
ORDERED_OBJ = gemma4_listener_attention_ordered.o
ATEN_OBJS = gemma4_listener_aten.o gemma4_listener_aten_norm.o
OBJS = $(CUDA_OBJS) $(ORDERED_OBJ) $(ATEN_OBJS)

all: $(TORCH_CONFIG) $(LIB_TARGET) $(TEST_TARGET)

$(TORCH_CONFIG): ../../tools/kernel_gates/export_torch_build_config.py
	python $< --output $@

%.o: %.cu generated_topology.h gemma4_listener.h gemma4_listener_kernels.cuh gemma4_listener_rms_exact.cuh listener_cuda_create_failure_log.cuh listener_create_failure_log.h sliding_repeat_capacity.h
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

gemma4_listener_attention_ordered.o: gemma4_listener_attention_ordered.cu gemma4_listener_attention_ordered.cuh $(TORCH_CONFIG)
	$(NVCC) $(ORDERED_NVCCFLAGS) -Ivendor/pytorch -Ivendor/cutlass $(TORCH_CXXFLAGS) -c $< -o $@

gemma4_listener_aten.o: gemma4_listener_aten.cpp gemma4_listener_aten.h gemma4_listener.h $(TORCH_CONFIG)
	$(CXX) $(CXXFLAGS) $(TORCH_CXXFLAGS) -c $< -o $@

gemma4_listener_aten_norm.o: gemma4_listener_aten_norm.cpp gemma4_listener_aten_norm.h $(TORCH_CONFIG)
	$(CXX) $(CXXFLAGS) $(TORCH_CXXFLAGS) -c $< -o $@

$(LIB_TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) -shared $(OBJS) -o $@ $(LDFLAGS) $(TORCH_LDFLAGS)

$(TEST_TARGET): gemma4_listener_test.cu $(LIB_TARGET)
	$(NVCC) $(NVCCFLAGS) $< -o $@ -L. -lgemma4_listener $(LDFLAGS) -Xlinker -rpath,.

clean:
	rm -f $(OBJS) $(LIB_TARGET) $(TEST_TARGET) $(TORCH_CONFIG)

.PHONY: all clean
