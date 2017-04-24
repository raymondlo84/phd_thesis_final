include Makefile.config

EXE=renderer

DEFINES += -D__STDC_CONSTANT_MACROS
DEFINES += -DUSE_YUV_TABLE

#ifeq "$(VERSION)" "RELEASE"
#LIBS += -fopenmp
#endif
#LIBS += -lboost_filesystem -lm -lpthread -ljpeg -lfftw3 -lserial
#LIBS += -lsfml-graphics -lsfml-window -lsfml-system
LIBS += -L/usr/lib/ `pkg-config --libs opencv` `pkg-config --cflags opencv` -lm

#for ffmpeg encoding...
#camera
#audio

OBJECTS += $(patsubst %.cpp,%.o,$(wildcard *.cpp))
OBJECTS += $(patsubst %.cu,%.o,$(wildcard *.cu))
CFLAGS += $(CUDA_INCLUDES) -I/usr/local/cuda/include 
NVCC_FLAGS += -I/usr/local/cuda/samples/common/inc/ 
LIBS += $(CUDA_LIB_DIRS) $(CUDA_LIBS)

OBJECTS += main.o

.cpp.o:
	$(CPP) $(CFLAGS) $(DEFINES) $(LIBS) -c $<
$(EXE): $(OBJECTS)
	$(CPP) $(CFLAGS) $(DEFINES) $^ -o $(EXE) $(LIBS)
chirplet_cuda.o: chirplet_cuda.cu chirplet_cuda.h
	$(NVCC) $(NVCC_FLAGS) $(DEFINES) -c $< -o $@
	
all: renderer

clean:
	rm -f *.o $(EXE)

