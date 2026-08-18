CXX      := g++
CXXFLAGS := -std=c++17 -O3 -ffp-contract=fast -fno-math-errno -ffinite-math-only \
            -DNDEBUG -DEIGEN_NO_DEBUG -D_USE_MATH_DEFINES
INCLUDES := -I. -I$(HOME)/eigen
TARGET   := test_sdopt
SRC      := src/test_main.cpp

HEADERS := $(wildcard **/*.h)

$(TARGET): $(SRC) $(HEADERS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)
