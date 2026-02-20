import time 

first_array = [10,50,100,200,500]
second_array = [10,50,100,200,500]

start = time.perf_counter()

for i in range(5):
    for j in range(5):
        print(first_array[i] * second_array[j])
        
end = time.perf_counter() 


print(f"Time of run:{end-start:.6f} seconds")