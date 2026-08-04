# Native RVV embedding engine (K3)

Hand-written FP32 RVV inference engines for two text embedding models, running directly on the
SpaceMIT K3's eight A100 AI harts -- no PyTorch, no ONNX Runtime, no sentence-transformers
anywhere in the inference path. See `PROGRESS.md` for full correctness numbers, benchmarks, and
the GEMM-kernel optimization history.

## Models

- `nomic-embed-text-v1.5` -- 768-dim, NomicBertModel (RoPE + SwiGLU, bidirectional, mean pooling)
- `mxbai-embed-large-v1` -- 1024-dim, standard BERT (absolute position embeddings + GELU, CLS pooling)

Both are drop-in reference-compatible with `willy@192.168.68.6:8100`'s `/embed_batch` and
`/embed_batch/mxbai` endpoints (same models, same "no L2 normalize" behavior).

## Regenerating the weight blobs

```sh
python3 -m venv .venv && source .venv/bin/activate
pip install torch transformers sentence-transformers einops safetensors scipy requests
python3 extract_nomic.py   # -> nomic.bin, nomic_manifest.json, nomic_vocab.txt
python3 extract_mxbai.py   # -> mxbai.bin, mxbai_manifest.json, mxbai_vocab.txt
```

Pulls fresh from the HF hub into this project's own `hf_cache/` -- never reads or writes the
graphify-embedder venv.

## Building on the K3 board

Native RISC-V codegen; build ON the board (no cross-compile toolchain set up for this project).

```sh
ssh root@192.168.68.24
cd ~/embed_native
gcc -march=rv64gcv_zfh_zvfh_xsmtvdotii -fno-tree-vectorize -O3 -DNOMIC_STANDALONE -o nomic_test nomic_rvv.c -lpthread -lm
gcc -march=rv64gcv_zfh_zvfh_xsmtvdotii -fno-tree-vectorize -O3 -shared -fPIC -o libnomic_rvv.so nomic_rvv.c -lpthread -lm
# same two commands with s/nomic/mxbai/ and -DMXBAI_STANDALONE
```

## Using it

From Python (tokenization + orchestration in Python, only the heavy math is native -- same
division of labor as `vision_api/ocr_native.py`):

```python
from bert_wordpiece import BertWordPieceTokenizer
from nomic_native import NomicEngine        # or: from mxbai_native import MxbaiEngine

tok = BertWordPieceTokenizer("nomic_vocab.txt")
eng = NomicEngine("nomic.bin")
ids = tok.encode("your text here")
embedding, wall_time_s = eng.embed(ids, nthreads=8)   # 768 floats, NOT L2-normalized
```

Or standalone, for benchmarking without Python overhead:

```sh
echo "101 7592 102" | ./nomic_test nomic.bin -nt 8              # prints the embedding
echo "101 7592 102" | ./nomic_test nomic.bin -nt 8 -bench 5      # repeated timing
echo "101 7592 102" | ./nomic_test nomic.bin -nt 8 -profile      # per-phase wall-time buckets
```

## Validating and benchmarking

```sh
python3 validate_native.py   # nomic vs live ryzen server + local pytorch oracle
python3 validate_mxbai.py    # same, for mxbai
python3 bench_nomic.py       # hart scaling (1/2/4/8) + vs ryzen, steady-state
python3 bench_mxbai.py       # same, for mxbai
```

All four run ON the K3 board (they load the native `.so` via ctypes and call the live ryzen
server over the LAN).
