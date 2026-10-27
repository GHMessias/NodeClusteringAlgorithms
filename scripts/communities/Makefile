CFLAGS = -O
CC = gcc
LIBS = -lm -lgsl -lgslcblas

estimate:estimate.o readgml.o
	$(CC) $(CFLAGS) -o estimate estimate.o readgml.o $(LIBS)

estimate.o:estimate.c network.h readgml.h Makefile
	$(CC) $(CFLAGS) -c estimate.c

readgml.o:readgml.c readgml.h network.h
	$(CC) $(CFLAGS) -c readgml.c
