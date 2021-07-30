CCFLAGS= -O3 -g
demo: demo.o mat.o 
	g++ demo.o mat.o $(CCFLAGS)

test: test.o mat.o
	g++ test.o mat.o $(CCFLAGS)

test.o: test.cc mat.h
	g++ -c test.cc $(CCFLAGS)

demo.o: demo.cc mat.h
	g++ -c demo.cc $(CCFLAGS)

mat.o: mat.cc mat.h
	g++ -c mat.cc $(CCFLAGS)

clean:
	rm *.o
