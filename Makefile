a.out: demo.o mat.o 
	g++ demo.o mat.o -O3

demo.o: demo.cc mat.h
	g++ -c demo.cc -O3

mat.o: mat.cc mat.h
	g++ -c mat.cc -O3

clean:
	rm *.o
