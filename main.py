import time 
import random
# multiplicacion de 500 elementos. Se supone que no le añada al tiempo 

first_array = [random.randint(1, 1000) for _ in range(500)]
second_array = [random.randint(1,1000) for _ in range(500)]

start = time.perf_counter()

for i in range(500):
    for j in range(500):
        print(first_array[i] * second_array[j])
        
end = time.perf_counter() 


print(f"Time of run:{end-start:.6f} seconds")