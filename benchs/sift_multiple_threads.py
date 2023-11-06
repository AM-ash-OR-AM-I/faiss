#!/usr/bin/env python3
"""Load data in parallel using threading or multiprocessing"""

from functools import wraps
import json
import os
import math
import argparse
import struct
import sys
import time
import multiprocessing
import logging
import threading
from typing import List
from threading import Thread
import numpy as np

from logging.handlers import RotatingFileHandler

os.makedirs("logs", exist_ok=True)

handlers = [
    RotatingFileHandler(
        "logs/sift_multiple_threads.log", maxBytes=1000000, backupCount=1
    ),
    logging.StreamHandler(),
]
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(name)s %(threadName)s : %(message)s",
    handlers=handlers,
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def calculate_chunk_size(file_path, num_chunks):
    total_size = os.path.getsize(file_path)  # Total size of the file in bytes
    chunk_size = math.ceil(total_size / num_chunks)  # Size of each chunk in bytes
    return chunk_size


def load_chunk(file_path, chunk_index, num_chunks):
    # Determine the dimension of the vectors
    with open(file_path, "rb") as f:
        dim = struct.unpack("i", f.read(4))[0]
    dtype = np.float32
    chunk_size = calculate_chunk_size(file_path, num_chunks) // (dim * dtype().itemsize)
    start_index = chunk_index * chunk_size
    end_index = start_index + chunk_size

    # Create a memory-mapped array for the chunk
    mmap = np.memmap(file_path, dtype=dtype, mode="r", shape=(end_index, dim))

    # Load the chunk into memory
    chunk = mmap[start_index:end_index]

    return chunk


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
    chunk_index: int,
    extract_file_path: str,
):
    """Load data in parallel"""
    logger.info("Loading data from chunk %s", chunk_index)
    base_dataset = load_chunk(f"{dir}/{file_prefix}.fvecs", chunk_index, CHUNKS)
    logger.info("Finished loading data from chunk %s", chunk_index)
    logger.info("Saving data from chunk %s", chunk_index)
    if os.path.exists(f"{extract_file_path}.txt"):
        os.remove(f"{extract_file_path}.txt")
    with open(f"{extract_file_path}.txt", "w") as file_obj:
        for vector in base_dataset:
            file_obj.write(str(vector.tolist()).replace(" ", "") + "\n")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description="Load data in parallel using threading or multiprocessing",
        usage="""
        LOAD SEQUENTIALLY:
        python3 benchs/sift_multiple_threads.py --chunks 100 --dataset-directory data/sift1M --file-prefix sift_base -o data/extracted_vectors

        LOAD IN PARALLEL USING MULTIPROCESSING:
        python3 benchs/sift_multiple_threads.py --chunks 100 --dataset-directory data/sift1M --file-prefix sift_base -o data/extracted_vectors --multiprocessing

        LOAD IN PARALLEL USING THREADING:
        python3 benchs/sift_multiple_threads.py --chunks 100 --dataset-directory data/sift1M --file-prefix sift_base -o data/extracted_vectors --threading
        """,
    )
    argparser.add_argument("-c", "--chunks", type=int, default=100)
    argparser.add_argument(
        "--multiprocessing",
        action=argparse.BooleanOptionalAction,
        default=False,
        dest="multiprocessing",
    )
    argparser.add_argument(
        "--threading",
        action=argparse.BooleanOptionalAction,
        default=False,
        dest="threading",
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

    process_time_map = {}

    # Save data in chunks
    file_prefix = args.file_prefix if args.file_prefix else ""
    # chunk_dataset(args.file_prefix, args.dir, CHUNKS)

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
                    (args.dir, file_prefix, i, f"{args.extract_file_path}_{i}")
                    for i in range(CHUNKS)
                ],
            )
            pool.close()
            pool.join()
            end = time.perf_counter()
            logger.info("Total time taken: %s", end - start)
            process_time_map["Total time taken"] = end - start

    elif args.threading:
        # Use threading to load data in parallel
        # Make sure threads are synchronized
        logger.info("Using threading")
        thread_pool: List[Thread] = []
        for i in range(CHUNKS):
            t = Thread(
                target=load_data,
                args=(args.dir, file_prefix, i, f"{args.extract_file_path}_{i}"),
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
    else:
        # Load data sequentially
        logger.info("Using sequential loading")
        start = time.perf_counter()
        for i in range(CHUNKS):
            load_data(args.dir, file_prefix, i, f"{args.extract_file_path}_{i}")
        end = time.perf_counter()
        logger.info("Total time taken: %s", end - start)
        process_time_map["Total time taken"] = end - start

    logger.info("time taken to load data: %s", process_time_map["Total time taken"])
    with open(
        f"logs/sift_{'multiprocessing' if args.multiprocessing else 'multiple_threads'}.json",
        "w",
    ) as file_obj:
        json.dump(process_time_map, file_obj, indent=4)
