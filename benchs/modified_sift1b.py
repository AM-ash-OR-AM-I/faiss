#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import print_function
import time
import numpy as np

import faiss
from datasets import load_sift, evaluate

import time

start = time.time()

print("load data")
xb, xq, xt, gt = load_sift(dir="siftsmall", file_prefix="siftsmall")
nq, d = xq.shape

for x in nq, d:
    print(str(list(x)).ljust(20), end=" ")
