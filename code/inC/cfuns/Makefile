name=cfuns

all: lib$(name).so

lib$(name).so: $(name).c $(name).h
	gcc -Wall -c -fPIC $(name).c
	gcc $(name).o -shared -Wl,-soname,lib$(name).so -o lib$(name).so

clean:
	rm -f lib$(name).so $(name).o *~

