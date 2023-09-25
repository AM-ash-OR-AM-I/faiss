# Results

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

## HNSW Benchmark

### Steps to reproduce

- Download `sift1M` using `curl -O ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz && tar -xvf sift.tar.gz`
- Rename `sift/` to `sift1M/`
- To run install [sift1M](../data/sift1M) dataset and run `python hnsw_benchmark.py 10` denoting the number of neighbors to search for k=10.

### Logs

- [Results](./results/hnsw_benchmark_k10.txt)

### CPU usage

- Can be recorded using `perf record python hnsw_benchmark.py 10` -> `perf report`
