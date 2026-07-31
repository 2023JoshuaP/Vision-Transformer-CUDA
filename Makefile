NVCC = nvcc
OPENCV_CFLAGS = `pkg-config --cflags opencv4`
OPENCV_LIBS = `pkg-config --libs opencv4`
NVCCFLAGS = -std=c++17 -O3 -arch=sm_89 -Iinclude $(OPENCV_CFLAGS)

SRC_DIR = src
INC_DIR = include
OBJ_DIR = obj
BIN_DIR = bin

SRCS = $(wildcard $(SRC_DIR)/*.cu) $(wildcard *.cu)

OBJS = $(patsubst $(SRC_DIR)/%.cu, $(OBJ_DIR)/%.o, $(wildcard $(SRC_DIR)/*.cu))
OBJS += $(patsubst %.cu, $(OBJ_DIR)/%.o, $(wildcard *.cu))

TARGET = $(BIN_DIR)/vit

.PHONY: all clean dirs run

all: dirs $(TARGET)

run: all
	./$(TARGET)

dirs:
	@mkdir -p $(OBJ_DIR)
	@mkdir -p $(BIN_DIR)

$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(OPENCV_LIBS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(OBJ_DIR)/%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean:
	rm -rf $(OBJ_DIR) $(BIN_DIR)
