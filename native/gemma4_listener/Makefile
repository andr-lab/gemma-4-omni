# Makefile — Builds libgemma4_listener.so and gemma4_listener_test
# Conforms to P4-NATIVE Implementation Plan §21

NVCC ?= /usr/local/cuda/bin/nvcc
CUDA_PATH ?= /usr/local/cuda

# This executor is certified only for the RTX 3090 target and deliberately
# does not enable --use_fast_math.
override ARCH = -gencode arch=compute_86,code=sm_86
NVCCFLAGS = -O3 $(ARCH) -lineinfo --compiler-options -fPIC -std=c++17 -I. -I$(CUDA_PATH)/include
LDFLAGS = -L$(CUDA_PATH)/lib64 -lcudart -lcublas

LIB_TARGET = libgemma4_listener.so
TEST_TARGET = gemma4_listener_test

SRCS = gemma4_listener.cu gemma4_listener_cache.cu
OBJS = $(SRCS:.cu=.o)

all: $(LIB_TARGET) $(TEST_TARGET)

%.o: %.cu generated_topology.h gemma4_listener.h gemma4_listener_kernels.cuh
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(LIB_TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) -shared $(OBJS) -o $@ $(LDFLAGS)

$(TEST_TARGET): gemma4_listener_test.cu $(LIB_TARGET)
	$(NVCC) $(NVCCFLAGS) $< -o $@ -L. -lgemma4_listener $(LDFLAGS) -Xlinker -rpath,.

clean:
	rm -f $(OBJS) $(LIB_TARGET) $(TEST_TARGET)

.PHONY: all clean
