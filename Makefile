CC = gcc
CFLAGS = -O2 -fPIC -Wall -Wextra -g -I./c_src/include
SRC = $(wildcard c_src/src/*.c)
OBJ = $(SRC:.c=.o)

OUTDIR = build
OUT = $(OUTDIR)/libmlp.so

all: $(OUT)

$(OUT): $(OBJ)
	mkdir -p $(OUTDIR)
	$(CC) -shared -lm -o $@ $^

clean:
	rm -rf $(OUTDIR) c_src/src/*.o

