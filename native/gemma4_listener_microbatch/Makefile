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

# Derive the ATen ABI and include/library paths from the Python runtime that
# will load this library. The microbatch candidate is intentionally buildable
# without the removed kernel-gates exporter or a hand-maintained config file.
PYTHON ?= python
TORCH_INCLUDE_FLAGS = $(shell $(PYTHON) -c "from torch.utils.cpp_extension import include_paths; print(' '.join('-I' + p for p in include_paths()))")
TORCH_LIBRARY_FLAGS = $(shell $(PYTHON) -c "from torch.utils.cpp_extension import library_paths; print(' '.join('-L' + p for p in library_paths()))")
TORCH_LIBRARY_DIR = $(shell $(PYTHON) -c "from torch.utils.cpp_extension import library_paths; print(library_paths()[0])")
TORCH_ABI_FLAG = -D_GLIBCXX_USE_CXX11_ABI=$(shell $(PYTHON) -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))")
TORCH_CXXFLAGS = $(TORCH_INCLUDE_FLAGS) $(TORCH_ABI_FLAG)
TORCH_LDFLAGS = $(TORCH_LIBRARY_FLAGS) -ltorch -ltorch_cpu -ltorch_cuda -lc10_cuda -lc10

LIB_TARGET = libgemma4_listener_microbatch.so
TEST_TARGET = gemma4_listener_microbatch_test

CUDA_SRCS = gemma4_listener.cu gemma4_listener_cache.cu gemma4_listener_rms_exact.cu
CUDA_OBJS = $(CUDA_SRCS:.cu=.o)
ORDERED_OBJ = gemma4_listener_attention_ordered.o
ATEN_OBJS = gemma4_listener_aten.o gemma4_listener_aten_norm.o
OBJS = $(CUDA_OBJS) $(ORDERED_OBJ) $(ATEN_OBJS)

all: $(LIB_TARGET) $(TEST_TARGET)

%.o: %.cu generated_topology.h gemma4_listener.h gemma4_listener_kernels.cuh gemma4_listener_rms_exact.cuh listener_cuda_create_failure_log.cuh listener_create_failure_log.h sliding_repeat_capacity.h full_repeat_capacity.h
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

gemma4_listener_attention_ordered.o: gemma4_listener_attention_ordered.cu gemma4_listener_attention_ordered.cuh
	$(NVCC) $(ORDERED_NVCCFLAGS) -I../gemma4_listener -I../gemma4_listener/vendor/cutlass $(TORCH_CXXFLAGS) -c $< -o $@

gemma4_listener_aten.o: gemma4_listener_aten.cpp gemma4_listener_aten.h gemma4_listener.h
	$(CXX) $(CXXFLAGS) $(TORCH_CXXFLAGS) -c $< -o $@

gemma4_listener_aten_norm.o: gemma4_listener_aten_norm.cpp gemma4_listener_aten_norm.h
	$(CXX) $(CXXFLAGS) $(TORCH_CXXFLAGS) -c $< -o $@

$(LIB_TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) -shared $(OBJS) -o $@ $(LDFLAGS) $(TORCH_LDFLAGS) -Xlinker -rpath -Xlinker $(TORCH_LIBRARY_DIR) -Xlinker -Bsymbolic-functions

gemma4_listener.o: gemma4_listener_microbatch.h gemma4_listener_microbatch_impl.inc

$(TEST_TARGET): gemma4_listener_test.cu $(LIB_TARGET)
	$(NVCC) $(NVCCFLAGS) $< -o $@ -L. -lgemma4_listener_microbatch $(LDFLAGS) $(TORCH_LDFLAGS) -Xlinker -rpath -Xlinker $(TORCH_LIBRARY_DIR) -Xlinker -rpath -Xlinker .

clean:
	rm -f $(OBJS) $(LIB_TARGET) $(TEST_TARGET)

cpu-contract-test:
	python ../../tests/kernel/test_native_listener_microbatch_binding.py

cpu-native-contract-test:
	$(CXX) -O2 -std=c++17 -I. test_history_batch_contract.cpp -o /tmp/gemma4_microbatch_history_contract
	/tmp/gemma4_microbatch_history_contract
	$(CXX) -O2 -std=c++17 -I. test_q4_gemv4_contract.cpp -o /tmp/gemma4_microbatch_q4_contract
	/tmp/gemma4_microbatch_q4_contract
	$(CXX) -O2 -std=c++17 -I. test_full_repeat_capacity.cpp -o /tmp/gemma4_microbatch_full_repeat_contract
	/tmp/gemma4_microbatch_full_repeat_contract

.PHONY: all clean cpu-contract-test cpu-native-contract-test
