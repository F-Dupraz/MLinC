CC = gcc
CFLAGS = -O2 -Wall

xor: examples/xor.c src/tensor.c src/nn.c src/node.c
	$(CC) $(CFLAGS) $^ -lm -o $@

clean:
	rm -f xor
