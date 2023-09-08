# Results of custom benchmarks

## ANN_SIFT_256K

```
load data
Loading siftsmall...done
train
PQ training on 25000 points, remains 0 points: training polysemous on centroids
add vectors to index
PQ baseline        0.100 ms per query, R@1 0.6300
Polysemous 64      0.089 ms per query, R@1 0.6300
Polysemous 62      0.072 ms per query, R@1 0.6300
Polysemous 58      0.047 ms per query, R@1 0.6300
Polysemous 54      0.046 ms per query, R@1 0.6300
Polysemous 50      0.025 ms per query, R@1 0.6300
Polysemous 46      0.020 ms per query, R@1 0.6100
Polysemous 42      0.018 ms per query, R@1 0.5800
Polysemous 38      0.019 ms per query, R@1 0.4600
Polysemous 34      0.017 ms per query, R@1 0.3100
Polysemous 30      0.020 ms per query, R@1 0.2500

Time taken = 33.288512229919434 secs
```

### TODO

- Profile the code to find the bottleneck
