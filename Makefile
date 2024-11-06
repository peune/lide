

INC_OPENCV = -I /opt/local/include/ 
LIB_OPENCV = -L /opt/local/lib -lopencv_core -lopencv_contrib -lopencv_calib3d -lopencv_features2d -lopencv_flann -lopencv_gpu -lopencv_highgui -lopencv_imgproc -lopencv_legacy -lopencv_ml -lopencv_nonfree -lopencv_objdetect -lopencv_photo -lopencv_stitching -lopencv_ts -lopencv_video -lopencv_videostab

CXX        = gcc
INCLUDE    = -I. 
CXXFLAGS   = $(INCLUDE) -Wall -g -fexceptions $(INC_OPENCV) -O3 # -DLINUX -O3
LDFLAGS    = -lm -lstdc++ -lc -L. $(LIB_OPENCV) 


# directory where all object files and other intermediat stuffs are stored
OBJSDIR = objs


# explicite creation of files in objs directory
$(OBJSDIR)/%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@


# by default the call 'make' does nothing
nothing:


# ignore files whose name is "all" or "clean"
.PHONY: all clean


# cleaning the object + bakup files
clean:
	rm -f $(OBJSDIR)/*.o *~


# list of object files
OBJS = \
	$(OBJSDIR)/he.o \
	$(OBJSDIR)/ahe.o \
	$(OBJSDIR)/dhe.o \
	$(OBJSDIR)/mhe.o \
	$(OBJSDIR)/esihe.o \
	$(OBJSDIR)/bpheme.o \
	$(OBJSDIR)/fhsabp.o \
	$(OBJSDIR)/hm.o \
	$(OBJSDIR)/lide_simple.o \
	$(OBJSDIR)/lide_mixture.o \
	$(OBJSDIR)/hegmm.o \
	$(OBJSDIR)/image_enhancement.o



test_enhancement: $(OBJSDIR)/test_enhancement.o $(OBJS)
	$(CXX) -o test_enhancement $(OBJSDIR)/test_enhancement.o $(OBJS) $(LDFLAGS) 
