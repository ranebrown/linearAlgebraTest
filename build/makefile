# makefile /bin/sh

# directories
SRCDIR = ../src
BUILDDIR = build
TESTDIR = ../test
INCDIR = ../inc

# Compiler to use
CC = gcc

# Include paths for header files
INCLUDES = -I $(INCDIR)

# Compiler flags
CFLAGS = -Wall -Wextra -g -Wno-implicit-int $(INCLUDES)

# Paths to required libraries -L flags
LFLAGS =  -L $(INCDIR)

# The specific libraries that project depends on -l flags
LIBS = -l meschach -l m -lcblas

# All source files
SRCS = $(wildcard $(SRCDIR)/*.c)

# All object files
OBJS := $(SRCS:$(SRCDIR)/%.c=%.o)

# name of executable
MAIN = exc

# make all
all: $(MAIN)

$(MAIN): $(OBJS)
	@$(CC) $(CFLAGS) -o $(MAIN) $(OBJS) $(LFLAGS) $(LIBS)	
	@echo "Compiling executable $(MAIN)..."
	@echo "Operation complete :)"

# Automatically builds all object files from source files
# -c option compiles but does not link (create object files)
# -o is output filename
$(OBJS): %.o : $(SRCDIR)/%.c
	@$(CC) $(CFLAGS) -c $< -o $@
	@echo "Compiling $@..."

# unit test setup
TESTFLAG = -lcheck

TESTSRCS = $(wildcard $(TESTDIR)/*.c) #add in any .c files from test dir

TESTOBJS := $(TESTSRCS:$(TESTDIR)/%.c=%.o) #convert names of all .c files to .o

MAINOBJS := $(filter-out main.o, $(OBJS)) #remove main.o from list of .o files (check doesn't use main)

$(TESTOBJS): %.o : $(TESTDIR)/%.c #compile but don't link .o files
	@$(CC) $(CFLAGS) -c $< -o $@
	@echo "Compiling $@..."

TESTEX = testexc # name of executable

$(TESTEX): $(TESTOBJS) #compile executable for test
	@$(CC) $(CFLAGS) $(TESTFLAG) -o $(TESTEX) $(TESTOBJS) $(MAINOBJS) $(LFLAGS) $(LIBS)
	@echo "Compiling executable $(TESTEX)..."
	@echo "Operation complete :)"
	@echo

# make and run unit tests
# makefile info: structure using $TESTEX here and above so target isn't rebuilt if no changes
# this is the same way main is set up
test: $(TESTEX)
	@./$(TESTEX)

# This is used in case there is a file named clean 
# @ suppresses command at shell
.PHONY: clean
clean:
	@if [ -e $(MAIN) ]; then rm $(MAIN); fi;
	@if [ -e $(TESTEX) ]; then rm $(TESTEX); fi;
	@if [ `find . -name "*.o" | head -1` ]; then rm *.o; fi; 
	@if [ `find . -name "*.txt" | head -1` ]; then rm *.txt; fi; 
	@echo "Cleaning..."
	@echo "Operation complete :)"

