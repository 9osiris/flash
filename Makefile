CXX = g++
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra
SRC = src/attn.cpp src/main.cpp
TEST_SRC = src/attn.cpp tests/test_flash.cpp

flash: $(SRC) src/attn.h
	$(CXX) $(CXXFLAGS) -o flash $(SRC)

test_flash: $(TEST_SRC) src/attn.h
	$(CXX) $(CXXFLAGS) -o test_flash $(TEST_SRC)

test: test_flash
	./test_flash

clean:
	rm -f flash test_flash

.PHONY: test clean
