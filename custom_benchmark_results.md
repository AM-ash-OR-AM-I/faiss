# Results of custom benchmarks

## Polysemous_ANN_128K benchmark

### Memory

- ![memory consumption](results/memory_consumption.png) from mprofile while running Polysemous_ANN_128K benchmark

### CPU usage

- ![CPU usage](results/cpu_usage.txt) using `perf record python benchs/bench_polysemous_sift1m.py` -> `perf report` 
  - Mainly the following command invocation were responsible for CPU usage:
  
  ```sh
  20.52%  python   _swigfaiss_avx2.cpython-311-x86_64-linux-gnu.so    [.] 0x00000000005ee040
  16.86%  python   _swigfaiss_avx2.cpython-311-x86_64-linux-gnu.so    [.] 0x00000000005ee06f
  15.49%  python   _swigfaiss_avx2.cpython-311-x86_64-linux-gnu.so    [.] 0x00000000005ee047
  12.76%  python   _swigfaiss_avx2.cpython-311-x86_64-linux-gnu.so    [.] 0x00000000005ee08d
  2.95%  python   _swigfaiss_avx2.cpython-311-x86_64-linux-gnu.so    [.] 0x00000000005ee0a4
  2.76%  python   _swigfaiss_avx2.cpython-311-x86_64-linux-gnu.so    [.] 0x00000000005ee09f
  .....
  ```

## TODO

- Check how FAISS works by running codes in tutorials
  
### Dependency

- Learn how to use `cmake` to build FAISS and numpy library to understand the inner-working.
