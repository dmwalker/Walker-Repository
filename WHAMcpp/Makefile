CC=g++
CFLAGS=-I. -O
DEPS = fileoptions.h BefeusTools.h WHAMfunctions.h 
OBJ = WHAM.o fileoptions.o BefeusTools.o WHAMfunctions.o

%.o: %.cpp $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

all: $(OBJ)
	$(CC) -o WHAM $^ $(CFLAGS)

.PHONY: clean

clean:
	rm -f $(OBJ) WHAM
