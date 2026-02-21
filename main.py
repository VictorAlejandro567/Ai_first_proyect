"""import time 
import random
# multiplicacion de 500 elementos. Se supone que no le añada al tiempo 

first_array = [random.randint(1, 1000) for _ in range(500)]
second_array = [random.randint(1,1000) for _ in range(500)]

start = time.perf_counter()

for i in range(500):
    for j in range(500):
        print(first_array[i] * second_array[j])
        
end = time.perf_counter() 


print(f"Time of run:{end-start:.6f} seconds")"""
import random
import statistics
import time


VECTOR_SIZES = [10, 50, 100, 200, 500]
MATRIX_SIZES = [10, 50, 100, 200, 500]
TRIALS = 10
SEED = 42


def make_vector(size: int, rng: random.Random) -> list[float]:
    return [rng.random() for _ in range(size)]


def make_matrix(size: int, rng: random.Random) -> list[list[float]]:
    return [[rng.random() for _ in range(size)] for _ in range(size)]

"Element-wise vector multiplication using pure Python (O(n))."
def vector_mul_python(a: list[float], b: list[float]) -> list[float]:
    if len(a) != len(b):
        raise ValueError("Vectors must have the same length.")

    out = [0.0] * len(a)
    for i in range(len(a)):
        out[i] = a[i] * b[i]
    return out

'Matrix multiplication using pure Python (O(n^3)).'
def matmul_python(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    n = len(a)
    if n == 0 or len(b) != n or any(len(row) != n for row in a) or any(len(row) != n for row in b):
        raise ValueError("Both matrices must be non-empty and square with the same dimensions.")

    result = [[0.0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for k in range(n):
            aik = a[i][k]
            for j in range(n):
                result[i][j] += aik * b[k][j]
    return result


def median_time(fn, trials: int = TRIALS) -> float:
    # Warm-up run (not measured).
    fn()

    samples = []
    for _ in range(trials):
        start = time.perf_counter()
        fn()
        end = time.perf_counter()
        samples.append(end - start)
    return statistics.median(samples)


def run_vector_benchmark() -> None:
    print("Pure Python vector element-wise multiplication (median seconds)")
    print("size,time_s")
    for size in VECTOR_SIZES:
        rng = random.Random(SEED + size)
        a = make_vector(size, rng)
        b = make_vector(size, rng)
        t = median_time(lambda: vector_mul_python(a, b))
        print(f"{size},{t:.8f}")
    print()


def run_matrix_benchmark() -> None:
    print("Pure Python matrix multiplication (median seconds)")
    print("size,time_s")
    for size in MATRIX_SIZES:
        rng = random.Random(SEED + 1000 + size)
        a = make_matrix(size, rng)
        b = make_matrix(size, rng)
        t = median_time(lambda: matmul_python(a, b))
        print(f"{size},{t:.8f}")


if __name__ == "__main__":
    run_vector_benchmark()
    run_matrix_benchmark()