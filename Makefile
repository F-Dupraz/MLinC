CC = gcc
CFLAGS = -O2 -Wall

xor: examples/xor.c src/mat.c src/nn.c src/activations.c
	$(CC) $(CFLAGS) $^ -lm -o $@

clean:
	rm -f xor
