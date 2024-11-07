
```
nvcc --shared -Xcompiler -fPIC -o attention.so attention.cu
python test.py "{prompt}"
```


TODOLIST
- ctype 없애기 : 불필요한 CPU-GPU 통신
- FlashAttention Implementation
