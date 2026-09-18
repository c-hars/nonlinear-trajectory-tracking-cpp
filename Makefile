ifeq ($(OS),Windows_NT)
    EIGEN_DIR ?= %USERPROFILE%/eigen
    TARGET    := test_sdopt.exe
    RM        := del /Q
    RMSUFFIX  := 2>nul
else
    EIGEN_DIR ?= $(HOME)/eigen
    TARGET    := test_sdopt
    RM        := rm -f
    RMSUFFIX  :=
endif

CXX      := g++
CXXFLAGS := -std=c++17 -O3 -DTEENSY_OPT_FASTER -ffp-contract=fast -fno-math-errno -ffinite-math-only \
            -DNDEBUG -DEIGEN_NO_DEBUG -D_USE_MATH_DEFINES
DEPFLAGS := -MMD -MP
INCLUDES := -I. -I$(EIGEN_DIR)
DEPFILE  := test_sdopt.d
SRC      := src/test_main.cpp

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(DEPFLAGS) $(INCLUDES) $< -o $@

-include $(DEPFILE)

run: $(TARGET)
	./$(TARGET)

clean:
	$(RM) $(TARGET) $(DEPFILE) $(RMSUFFIX)