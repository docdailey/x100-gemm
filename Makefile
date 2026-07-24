# x100-gemm — GEMM for the SpaceMIT X100 (RISC-V RVA23): RVV 1.0 + IME.
#
# Native build on the board (gcc 15, RVV):   make
# Enable the IME int8 path (needs clang + SpaceMIT xsmtvdot):
#     make CC=clang IME=1
#
# The RVV path builds with mainline gcc/clang. The IME (vmadot) path is gated
# behind X100_HAVE_IME and needs a toolchain that assembles smt.vmadot.

CC      ?= gcc
STD      = -std=gnu11
WARN     = -Wall -Wextra
OPT      = -O3 -ffast-math -funroll-loops
# RVV 1.0 + fp16 vectors. (X100: rv64gcv + zvfh; VLEN=256.)
ARCH    ?= rv64gcv_zvfh
INC      = -Iinclude
CFLAGS   = $(STD) $(WARN) $(OPT) -march=$(ARCH) $(INC)
LDLIBS   = -lm

ifeq ($(IME),1)
  # SpaceMIT IME: X60 subset = xsmtvdot ; A100 = xsmtvdotii. Adjust per core.
  ARCH   := $(ARCH)_xsmtvdot
  CFLAGS += -DX100_HAVE_IME
endif

SRC = src/x100_caps.c src/gemm_ref.c src/gemm_rvv_f32.c src/gemm_rvv_f16.c src/gemm_ime_i8.c
OBJ = $(SRC:.c=.o)

BIN = gemm_bench

.PHONY: all caps clean asm
all: $(BIN)

$(BIN): bench/bench.c $(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDLIBS)
	@echo "built ./$(BIN)   (arch=$(ARCH), CC=$(CC))"

%.o: %.c include/x100_gemm.h
	$(CC) $(CFLAGS) -c -o $@ $<

# quick capability print
caps: $(BIN)
	./$(BIN) 8 8 8 1 | sed -n '1,8p'

# dump the RVV micro-kernel asm to eyeball vsetvli/vfmacc
asm:
	$(CC) $(CFLAGS) -S -o - src/gemm_rvv_f32.c | sed -n '/gemm_rvv_f32:/,/\.size/p'

clean:
	rm -f $(OBJ) $(BIN)
