CXX = g++
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra
SIMD_FLAGS = -mavx2 -mfma
SRC = src/attn.cpp src/main.cpp
TEST_SRC = src/attn.cpp tests/test_flash.cpp

flash: $(SRC) src/attn.h src/simd.h
	$(CXX) $(CXXFLAGS) -o flash $(SRC)

test_flash: $(TEST_SRC) src/attn.h src/simd.h
	$(CXX) $(CXXFLAGS) -o test_flash $(TEST_SRC)

test: test_flash
	./test_flash

# avx2 build: same code, the dot and axpy helpers in src/simd.h
# take the vector path. needs a cpu with avx2 and fma.
flash_simd: $(SRC) src/attn.h src/simd.h
	$(CXX) $(CXXFLAGS) $(SIMD_FLAGS) -o flash_simd $(SRC)

test_flash_simd: $(TEST_SRC) src/attn.h src/simd.h
	$(CXX) $(CXXFLAGS) $(SIMD_FLAGS) -o test_flash_simd $(TEST_SRC)

simd: test_flash_simd flash_simd
	./test_flash_simd
	./flash_simd verify 512 64
	./flash_simd bench 512 64
	./flash_simd bench 2048 64

clean:
	rm -f flash test_flash flash_simd test_flash_simd

.PHONY: test clean simd
