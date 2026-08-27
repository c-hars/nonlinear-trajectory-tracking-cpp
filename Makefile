CXX      := g++
CXXFLAGS := -std=c++17 -O3 -ffp-contract=fast -fno-math-errno -ffinite-math-only \
            -DNDEBUG -DEIGEN_NO_DEBUG -D_USE_MATH_DEFINES
DEPFLAGS := -MMD -MP
EIGEN_DIR ?= $(HOME)/eigen
INCLUDES := -I. -I$(EIGEN_DIR)
TARGET   := test_sdopt
SRC      := src/test_main.cpp

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) $(INCLUDES) $< -o $@

-include $(TARGET).d

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET) $(TARGET).d