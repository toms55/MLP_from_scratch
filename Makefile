CC = gcc

CFLAGS = -O3 -march=native -funroll-loops -fomit-frame-pointer -flto -fPIC -I./c_src/include
LDFLAGS = -flto -lm -lopenblas

SRC = $(wildcard c_src/src/*.c)
OBJ = $(SRC:.c=.o)

OUTDIR = build
LIBOUT = $(OUTDIR)/libmlp.so

# Test program
TEST_SRC = tests/main.c
TEST_OUT = $(OUTDIR)/test_matrix

all: $(LIBOUT)

# Build the shared library
$(LIBOUT): $(OBJ)
	mkdir -p $(OUTDIR)
	$(CC) -shared -lm -o $@ $^

# Build test program
test: $(OBJ) $(TEST_SRC)
	mkdir -p $(OUTDIR)
	$(CC) $(CFLAGS) -o $(TEST_OUT) $(TEST_SRC) $(OBJ) -lm

# Run test program
run: test
	./$(TEST_OUT)

clean:
	rm -rf $(OUTDIR) c_src/src/*.o
