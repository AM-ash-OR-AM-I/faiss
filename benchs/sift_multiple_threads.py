#!/usr/bin/env python3
"""Load data in parallel using threading or multiprocessing"""

from functools import wraps
import json
import os
import argparse
import sys
import time
import multiprocessing
import logging
import threading
from typing import List
from threading import Thread
import numpy as np


os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    filename="logs/output.log",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def ivecs_read(fname):
    a = np.fromfile(fname, dtype="int32")
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


def fvecs_read(fname):
    return ivecs_read(fname).view("float32")


def load_model(dir: str = "sift1M", file_prefix="", numpy=True):
    print(f"Loading {dir}/{file_prefix}...", end="", file=sys.stderr)
    if not file_prefix:
        raise ValueError("file not found")
    if numpy:
        xb = np.load(f"{dir}/{file_prefix}.npy")
    else:
        xb = fvecs_read(f"{dir}/{file_prefix}.fvecs")
    print("done", file=sys.stderr)
    return xb


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
def load_data(
    dir: str,
    file_prefix: str,
    extract_file_path: str,
):
    """Load data in parallel"""
    base_dataset = load_model(dir, file_prefix)
    if os.path.exists(f"{extract_file_path}.txt"):
        os.remove(f"{extract_file_path}.txt")
    with open(f"{extract_file_path}.txt", "w") as file_obj:
        for vector in base_dataset:
            file_obj.write(str(vector.tolist()).replace(" ", "") + "\n")


def chunk_dataset(file_prefix: str, dir: str, chunks: int):
    """Chunk data into multiple files"""
    xb = load_model(dir, file_prefix, numpy=False)
    nb, d = xb.shape
    chunk_size = nb // chunks

    for i in range(chunks):
        chunk_name = f"{dir}/{file_prefix}_{i}"
        if os.path.exists(chunk_name):
            continue
        np.save(
            chunk_name,
            xb[i * chunk_size : (i + 1) * chunk_size],
        )


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="Load data in parallel using threading or multiprocessing",
        usage="""
        python3 benchs/sift_multiple_threads.py --chunks 100 --multiprocessing --dataset-directory data/sift1M --file-prefix sift_base -o data/extracted_vectors.txt
        """,
    )
    argparser.add_argument("-c", "--chunks", type=int, default=100)
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
        required=True,
        default="data/sift1M",
        dest="dir",
        help="Path to dataset for SIFT1B/1M dataset",
    )
    argparser.add_argument(
        "-prefix",
        "--file-prefix",
        type=str,
        default="sift",
        dest="file_prefix",
        help="Prefix for SIFT1B/1M dataset files",
    )
    argparser.add_argument(
        "-o",
        "--extracted-file-path",
        type=str,
        required=True,
        default="data/extract_multiple_texts.txt",
        dest="extract_file_path",
        help="Path to save extracted file",
    )
    args = argparser.parse_args()
    CHUNKS = args.chunks

    logger.info("Loading data")

    process_time_map = {}

    # Save data in chunks
    file_prefix = args.file_prefix if args.file_prefix else ""
    chunk_dataset(args.file_prefix, args.dir, CHUNKS)

    # Load data from chunks
    if args.multiprocessing:
        # Use multiprocessing to load data in parallel
        # Multiprocessing is much more faster than threading
        logger.info("Using multiprocessing")
        with multiprocessing.Pool(processes=CHUNKS) as pool:
            start = time.perf_counter()
            pool.starmap(
                load_data,
                [
                    (args.dir, f"{file_prefix}_{i}", f"{args.extract_file_path}_{i}")
                    for i in range(CHUNKS)
                ],
            )
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
        for i in range(CHUNKS):
            t = Thread(
                target=load_data,
                args=(args.dir, f"{file_prefix}_{i}", f"{args.extract_file_path}_{i}"),
                name=f"Thread-{i}",
            )
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
