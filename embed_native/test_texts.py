"""Deterministic test-text set shared by tokenizer validation, correctness validation, and
benchmarking. Varied lengths: single words -> near-max-length passages. Real English text
(a mix of technical/general prose), not lorem ipsum.
"""

TEST_TEXTS = [
    "hello",
    "GEMM",
    "RISC-V",
    "The quick brown fox jumps over the lazy dog.",
    "search_query: what is the capital of France?",
    "search_document: Paris is the capital and most populous city of France.",
    "Embeddings map discrete tokens into a continuous vector space where semantic "
    "similarity corresponds to geometric proximity.",
    "The SpaceMIT K3 board exposes eight A100 RISC-V harts, each with a 1024-bit vector "
    "register file (VLEN=1024) and an IME-2 matrix/dot-product extension. Four of the "
    "eight harts (8, 10, 12, 14) are never contended for IME-2 work; the other four "
    "(9, 11, 13, 15) share IME-2 hardware pairwise with a primary partner, though plain "
    "RVV vector FMA work has been measured to scale cleanly across all eight harts with "
    "no such contention.",
    (
        "In mathematics, a matrix is a rectangular array of numbers, symbols, or expressions, "
        "arranged in rows and columns, which is used to represent a mathematical object or a "
        "property of such an object. For example, the dimensions of the matrix below are 2 x 3 "
        "(read 'two by three'), because there are two rows and three columns. The individual "
        "items, or entries, of a matrix are often denoted by a variable with two subscripts. "
        "Matrices are commonly related to linear algebra. Notable exceptions include incidence "
        "matrices and adjacency matrices in graph theory. This article focuses on matrices "
        "related to linear algebra, and, unless otherwise specified, all matrices represent "
        "linear maps or may be viewed as such. Square matrices, matrices with the same number "
        "of rows and columns, play a major role in matrix theory. Square matrices of a given "
        "dimension form a noncommutative ring, which is one of the most common examples of a "
        "noncommutative ring. The determinant of a square matrix is a number associated with "
        "the matrix, which is fundamental for the study of a square matrix; for example, a "
        "square matrix is invertible if and only if it has a nonzero determinant, and the "
        "eigenvalues of a square matrix are the roots of a polynomial determinant."
    ),
    "a",
    "12345 67890 !@#$% test-case_underscore CamelCase naïve café résumé "
    "über façade coöperate 日本語 中文测试",
    "",
    "   ",
    "Bill's project runs a hand-written RVV inference engine -- no PyTorch, no ONNX Runtime, "
    "no XLA -- directly on the K3's eight A100 AI harts, and it validates bit-exact against "
    "a real oracle before anyone trusts a single benchmark number.",
]
