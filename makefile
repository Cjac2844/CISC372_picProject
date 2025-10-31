
image:image.c image.h
	gcc -g image.c -o image -lm
clean:
	rm -f image output.png
