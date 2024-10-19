nvcc --shared -Xcompiler -fPIC -o attention.so attention.cu'''
python test.py "{prompt}"
