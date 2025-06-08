# Makefile for braiNNlet project

CXX = g++
CXXFLAGS = -std=c++20 -Wall -Wextra -O2
BUILD_DIR = cmake-build-debug

# Default target builds everything
all: build cli

# Create build directory and run cmake
cmake-configure:
	mkdir -p $(BUILD_DIR)
	cd $(BUILD_DIR) && cmake ..

# Build all targets
build: cmake-configure
	cd $(BUILD_DIR) && make

# Build specific targets
$(BUILD_DIR)/braiNNlet: cmake-configure
	cd $(BUILD_DIR) && make braiNNlet

$(BUILD_DIR)/braiNNlet-cli: cmake-configure
	cd $(BUILD_DIR) && make braiNNlet-cli

$(BUILD_DIR)/test_braiNNlet: cmake-configure
	cd $(BUILD_DIR) && make test_braiNNlet

# Convenience targets
main: $(BUILD_DIR)/braiNNlet

cli: $(BUILD_DIR)/braiNNlet-cli

test-build: $(BUILD_DIR)/test_braiNNlet

# Run targets
run: $(BUILD_DIR)/braiNNlet
	cd $(BUILD_DIR) && ./braiNNlet

run-cli: $(BUILD_DIR)/braiNNlet-cli
	cd $(BUILD_DIR) && ./braiNNlet-cli

test: $(BUILD_DIR)/test_braiNNlet
	cd $(BUILD_DIR) && ./test_braiNNlet

# Format all C++ files
format:
	find . -name "*.cpp" -o -name "*.hpp" | grep -E "(src|cli|tests)" | xargs clang-format -i

# Clean build directory
clean:
	rm -rf $(BUILD_DIR)

# Show help
help:
	@echo "braiNNlet Neural Network Library - Build System"
	@echo "=============================================="
	@echo ""
	@echo "Targets:"
	@echo "  all       - Build everything (library, main demo, CLI)"
	@echo "  build     - Build all targets using CMake"
	@echo "  main      - Build main demo executable"
	@echo "  cli       - Build CLI application"
	@echo "  test-build - Build test executable"
	@echo ""
	@echo "Run targets:"
	@echo "  run       - Run the main demo"
	@echo "  run-cli   - Run the CLI application"
	@echo "  test      - Run the test suite"
	@echo ""
	@echo "Utilities:"
	@echo "  format    - Format all C++ source files"
	@echo "  clean     - Clean build directory"
	@echo "  help      - Show this help message"

.PHONY: all build main cli test-build run run-cli test format clean help cmake-configure gui-deps gui-dev gui-start gui-build