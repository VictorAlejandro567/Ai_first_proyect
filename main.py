import time
import random
import statistics as stats
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Configuration
# -----------------------------

SEED = 12345
random.seed(SEED)
np.random.seed(SEED)

DTYPE = np.float64

VECTOR_SIZES = [10, 50, 100, 200, 500]
MATRIX_SIZES = [10, 50, 100, 200, 500]
TRIALS = 10


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

vec_rows = []

for n in VECTOR_SIZES:

    # Creates arrays OUTSIDE timed region
    a_py = [float(random.random()) for _ in range(n)]
    b_py = [float(random.random()) for _ in range(n)]

    a_np = np.array(a_py, dtype=DTYPE)
    b_np = np.array(b_py, dtype=DTYPE)

    # Correctness verification
    out_py = py_vector_multiply(a_py, b_py) #Pure Python element-wise multiply
    out_np = a_np * b_np #Numpy's element-wise multiply
    assert np.allclose(np.array(out_py, dtype=DTYPE), out_np)

    # Timing only multiplication
    t_py = median_time(lambda: py_vector_multiply(a_py, b_py))
    t_np = median_time(lambda: a_np * b_np)

    speedup = t_py / t_np if t_np > 0 else float("nan")

    vec_rows.append({
        "n": n,
        "t_python_s": t_py,
        "t_numpy_s": t_np,
        "speedup": speedup
    })

    print(f"Size {n:4d} -> Python: {t_py:.6e}s | NumPy: {t_np:.6e}s | Speedup: {speedup:.2f}x")

df_vec = pd.DataFrame(vec_rows)


# =============================
# MATRIX BENCHMARK
# =============================

print("\n--- MATRIX MULTIPLICATION ---")

mat_rows = []

for n in MATRIX_SIZES:

    print(f"\nRunning {n}x{n}...")

    # Creates matrices OUTSIDE timed region
    A_py = [[float(random.random()) for _ in range(n)] for _ in range(n)]
    B_py = [[float(random.random()) for _ in range(n)] for _ in range(n)]

    A_np = np.array(A_py, dtype=DTYPE)
    B_np = np.array(B_py, dtype=DTYPE)

    # Correctness verification
    C_py = py_matrix_multiply(A_py, B_py) #Pure Python matrix multiplication
    C_np = A_np @ B_np #Numpy's matrix multiplication
    assert np.allclose(np.array(C_py, dtype=DTYPE), C_np)

    # Timing only multiplication
    t_py = median_time(lambda: py_matrix_multiply(A_py, B_py))
    t_np = median_time(lambda: A_np @ B_np)

    speedup = t_py / t_np if t_np > 0 else float("nan")

    mat_rows.append({
        "n": n,
        "t_python_s": t_py,
        "t_numpy_s": t_np,
        "speedup": speedup
    })

    print(f"Size {n:4d}x{n:4d} -> Python: {t_py:.6e}s | NumPy: {t_np:.6e}s | Speedup: {speedup:.2f}x")

df_mat = pd.DataFrame(mat_rows)


# =============================
# RESULTS TABLES
# =============================

print("\nVector Results:")
print(df_vec.to_string(index=False))

print("\nMatrix Results:")
print(df_mat.to_string(index=False))


# =============================
# REQUIRED PLOT
# =============================

plt.figure()
plt.plot(df_mat["n"], df_mat["t_python_s"], marker="o", label="Pure Python")
plt.plot(df_mat["n"], df_mat["t_numpy_s"], marker="o", label="NumPy")
plt.xlabel("Matrix Dimension (n x n)")
plt.ylabel("Median Runtime (seconds)")
plt.title("Matrix Multiplication Runtime vs Dimension")
plt.yscale("log")
plt.grid(True, which="both")
plt.legend()
plt.show()


# =============================
# OPTIONAL SPEEDUP PLOT
# =============================

plt.figure()
plt.plot(df_mat["n"], df_mat["speedup"], marker="o")
plt.xlabel("Matrix Dimension (n x n)")
plt.ylabel("Speedup = T_python / T_numpy")
plt.title("Speedup of NumPy over Pure Python")
plt.grid(True, which="both")
plt.show()



plt.figure()
plt.plot(df_vec["n"], df_vec["t_python_s"], marker="o", label="Pure Python")
plt.plot(df_vec["n"], df_vec["t_numpy_s"], marker="o", label="NumPy")
plt.xlabel("Vector Length (n)")
plt.ylabel("Median Runtime (seconds)")
plt.title("Vector Element-wise Multiplication Runtime vs Size")
plt.yscale("log")
plt.grid(True, which="both")
plt.legend()
plt.show()