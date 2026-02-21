# import time 
# import random
# multiplicacion de 500 elementos. Se supone que no le añada al tiempo 

# first_array = [random.randint(1, 1000) for _ in range(500)]
# second_array = [random.randint(1,1000) for _ in range(500)]

# start = time.perf_counter()

# for i in range(500):
#     for j in range(500):
#         print(first_array[i] * second_array[j])

# end = time.perf_counter() 

# print(f"Time of run:{end-start:.6f} seconds")
import time
import random
import statistics as stats

# -----------------------------
# Configuration
# -----------------------------

SEED = 12345
random.seed(SEED)

VECTOR_SIZES = [10, 50, 100, 200, 500]
MATRIX_SIZES = [10, 50, 100, 200, 500]
TRIALS = 15


# -----------------------------
# Vector element-wise multiply
# -----------------------------

def py_vector_multiply(a, b):
    n = len(a)
    result = [0.0] * n
    for i in range(n):
        result[i] = a[i] * b[i]
    return result


# -----------------------------
# True matrix multiplication
# -----------------------------

def py_matrix_multiply(A, B):
    n = len(A)
    C = [[0.0] * n for _ in range(n)]

    for i in range(n):
        Ai = A[i]
        Ci = C[i]
        for k in range(n):
            aik = Ai[k]
            Bk = B[k]
            for j in range(n):
                Ci[j] += aik * Bk[j]

    return C


# -----------------------------
# Timing helper
# -----------------------------

def median_time(function, trials=TRIALS, warmup=True):
    if warmup:
        function()

    times = []
    for _ in range(trials):
        start = time.perf_counter()
        function()
        end = time.perf_counter()
        times.append(end - start)

    return stats.median(times)


# =============================
# VECTOR BENCHMARK
# =============================

print("\n--- VECTOR ELEMENT-WISE MULTIPLICATION ---")

for n in VECTOR_SIZES:
    a = [random.random() for _ in range(n)]
    b = [random.random() for _ in range(n)]

    t = median_time(lambda: py_vector_multiply(a, b), TRIALS)
    print(f"Size {n:4d} -> Median Time: {t:.6f} seconds")


# =============================
# MATRIX BENCHMARK
# =============================

print("\n--- MATRIX MULTIPLICATION ---")

for n in MATRIX_SIZES:
    print(f"\nRunning {n}x{n}...")

    A = [[random.random() for _ in range(n)] for _ in range(n)]
    B = [[random.random() for _ in range(n)] for _ in range(n)]

    t = median_time(lambda: py_matrix_multiply(A, B), TRIALS)
    print(f"Size {n:4d}x{n:4d} -> Median Time: {t:.6f} seconds")