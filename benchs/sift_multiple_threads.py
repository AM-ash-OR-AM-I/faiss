#!/usr/bin/env python3
"""Load data in parallel using threading or multiprocessing"""

from functools import wraps
import json
import os
import argparse
import time
import multiprocessing
import logging
import threading
from typing import List
from threading import Thread
import numpy as np

from datasets import load_sift


os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    filename="logs/output.log",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def thread_time_decorator(thread):
    """
    Decorator to capture thread execution time
    """

    @wraps(thread)
    def wrapper(*args, **kwargs):
        _start = time.perf_counter()
        thread(*args, **kwargs)
        _end = time.perf_counter()
        threading.current_thread().thread_duration = _end - _start

    return wrapper


@thread_time_decorator
def load_data(training_dataset):
    """Load data in parallel"""

    for vector in training_dataset:
        print(str(list(vector)).replace(" ", ""))


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="Load data in parallel using threading or multiprocessing",
        usage="""
        python3 benchs/sift_multiple_threads.py --multiprocessing --processes 25 --dataset-directory data/sift1B --prefix sift
        """,
    )
    argparser.add_argument("-n", "--processes", type=int, default=25)
    argparser.add_argument(
        "--multiprocessing",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="multiprocessing",
    )
    argparser.add_argument(
        "-dir",
        "--dataset-directory",
        type=str,
        default="data/sift1M",
        dest="data",
        help="Path to dataset for SIFT1B/1M dataset",
    )
    argparser.add_argument(
        "-prefix",
        "--file_prefix",
        type=str,
        default="sift",
        dest="file_prefix",
        help="Prefix for SIFT1B/1M dataset files",
    )
    args = argparser.parse_args()
    NUMBER_OF_PROCESSES = args.processes

    logger.info("Loading data")
    xb, xq, xt, gt = load_sift(dir=args.data, file_prefix=args.file_prefix)
    nb, d = xb.shape
    logger.debug("xb: %s", xb.shape)

    # Divide xb into number of processes
    process_time_map = {}
    xb_splitted = np.array_split(xb, NUMBER_OF_PROCESSES)
    logger.debug("xb_splitted: %s", len(xb_splitted))
    if args.multiprocessing:
        # Use multiprocessing to load data in parallel
        # Multiprocessing is much more faster than threading
        logger.info("Using multiprocessing")
        with multiprocessing.Pool(processes=NUMBER_OF_PROCESSES) as pool:
            start = time.perf_counter()
            pool.map(load_data, xb_splitted)
            pool.close()
            pool.join()
            end = time.perf_counter()
            logger.info("Total time taken: %s", end - start)
            process_time_map["Total time taken"] = end - start

    else:
        # Use threading to load data in parallel
        # Make sure threads are synchronized
        logger.info("Using threading")
        thread_pool: List[Thread] = []
        for i in range(NUMBER_OF_PROCESSES):
            t = Thread(target=load_data, args=(xb_splitted[i],), name=f"Thread_{i}")
            thread_pool.append(t)
            t.start()

        for t in thread_pool:
            t.join()
            logger.info("Thread %s took %s seconds", t.name, t.thread_duration)
            process_time_map[t.name] = t.thread_duration
        process_time_map["Total time taken"] = sum(process_time_map.values())
        process_time_map["Total No of threads"] = len(thread_pool)
        process_time_map["Average thread time"] = process_time_map[
            "Total No of threads"
        ] / len(thread_pool)
        logger.info("Total No of threads used: %s", len(thread_pool))
    logger.info("time taken to load data: %s", process_time_map["Total time taken"])
    with open(
        f"logs/sift_{'multiprocessing' if args.multiprocessing else 'multiple_threads'}.json",
        "w",
    ) as file_obj:
        json.dump(process_time_map, file_obj, indent=4)
