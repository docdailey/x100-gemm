# x100-gemm — GEMM for the SpaceMIT X100 (RISC-V RVA23): RVV 1.0 + IME.
#
# Native build on the board (Bianbu gcc 15.2, RVV):   make
# Enable the IME int8 (vmadot) path — the board's gcc supports it natively:
#     make IME=1
#
# CONFIRMED: Bianbu gcc 15.2.0 assembles+runs smt.vmadot with -march=...xsmtvdotii
# (no clang / SpaceMIT toolchain needed). The IME path is gated behind X100_HAVE_IME.

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
  # SpaceMIT IME: X100 uses xsmtvdotii (confirmed: gcc assembles+runs smt.vmadot).
  ARCH   := $(ARCH)_xsmtvdotii
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
