import json
import time
import argparse
import threading
from tqdm import tqdm
from queue import Queue, PriorityQueue
from concurrent.futures import ThreadPoolExecutor, as_completed

import sglang as sgl
from sglang import set_default_backend, RuntimeEndpoint

from aw_engine.backends import SGLangBackend


def common_llm_call(persona_name, step, prompt, max_tokens, stop, instrument=False):

    if instrument:
        return SGLangBackend.generate(prompt,
                                      max_tokens=max_tokens,
                                      step=step,
                                      stop=stop,
                                      trace_id=f"{persona_name}:{step}")
    else:
        set_default_backend(RuntimeEndpoint("http://localhost:30000"))

        # note a cumstomized SGlang needed to incorporate priority scheduling
        if not priority:
            priority = step

        @sgl.function
        def func_wrapper(s, prompt):
            s += prompt
            s += sgl.gen("response", max_tokens=max_tokens, stop=stop)

        response = func_wrapper.run(prompt)
        return response


def duplicate_traces(dependency_adjacency, persona_calls, duplication):
    duplicated_dag_adjacency = {}
    duplicated_dag_adjacency_reverse = {}
    duplicated_global_dict = {}

    for i in range(duplication):
        for node in dependency_adjacency["forward"]:
            assert node in dependency_adjacency["reverse"]
            assert node in persona_calls
            persona, step = node.split(":")
            new_key = f"{persona}_{i}:{step}"
            duplicated_dag_adjacency[new_key] = [
                f"{n.split(':')[0]}_{i}:{n.split(':')[1]}" for n in dependency_adjacency["forward"][node]
            ]
            duplicated_dag_adjacency_reverse[new_key] = [
                f"{n.split(':')[0]}_{i}:{n.split(':')[1]}" for n in dependency_adjacency["reverse"][node]
            ]
            duplicated_global_dict[new_key] = persona_calls[node]

    return duplicated_dag_adjacency, duplicated_dag_adjacency_reverse, duplicated_global_dict


def bench_oracle(dependency_adjacency, persona_calls, rate_limit, duplication=1, priority=None, instrument=False):

    dag_adjacency, dag_adjacency_reverse, persona_calls = duplicate_traces(dependency_adjacency, persona_calls,
                                                                           duplication)

    pq = PriorityQueue()
    dag_lock = threading.Lock()

    def enqueue(work):
        if not priority:
            pq.put((0, work))
        elif priority == "step":
            pq.put((int(work.split(":")[1]), work))
        elif priority == "predefined":
            assert "priority" in persona_calls[work]
            pq.put((persona_calls[work]["priority"], work))
        else:
            raise NotImplementedError

    # Initialize the queue with nodes that have no reverse dependencies
    for work, dependencies in dag_adjacency_reverse.items():
        if not dependencies:
            enqueue(work)

    progress_bar = tqdm(total=sum([len(persona_calls[p]["funcs"]) for p in persona_calls]))

    def worker():
        while True:
            work = pq.get()
            if work is None:
                # finishing up benchmark
                pq.put(None)
                break
            _, work = work
            assert work in dag_adjacency
            assert work in dag_adjacency_reverse
            assert not dag_adjacency_reverse[work]
            persona, step = work.split(":")
            for config in persona_calls[work]["funcs"]:
                common_llm_call(persona,
                                int(step),
                                config["prompt"],
                                config["max_tokens"],
                                config["stop"],
                                instrument=instrument)
                progress_bar.update(1)
            with dag_lock:
                for blocked in dag_adjacency.pop(work):
                    dag_adjacency_reverse[blocked].remove(work)
                    if not dag_adjacency_reverse[blocked]:
                        enqueue(blocked)

            if dag_adjacency == {}:
                pq.put(None)
                return

    start_time = time.time()
    num_initial_requests = pq.qsize()
    with ThreadPoolExecutor(max_workers=rate_limit) as executor:
        futures = [executor.submit(worker) for _ in range(num_initial_requests)]
        for future in as_completed(futures):
            future.result()
    print(f"Time taken for the benchmark: {time.time() - start_time}")


def bench_limit(persona_calls, duplication=1):

    works = [w for _ in range(duplication) for i, j in persona_calls.items() for w in j["funcs"]]
    progress_bar = tqdm(total=len(works))

    def worker(w):
        common_llm_call("", 0, w["prompt"], w["max_tokens"], w["stop"])
        progress_bar.update(1)

    start_time = time.time()
    with ThreadPoolExecutor(max_workers=256) as executor:
        futures = [executor.submit(worker, w) for w in works]
        for future in as_completed(futures):
            future.result()
    print(f"Time taken for the benchmark: {time.time() - start_time}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--priority', type=str, default=None, choices=[None, "step", "predefined"])
    parser.add_argument('--num-agents', type=int, default=25)
    parser.add_argument('--rate-limit', type=int, default=256, help="this should be changed according to num of GPUs")
    parser.add_argument('--dag', type=str, default="dependency_adjacency.json")
    parser.add_argument('--trace', type=str, default="persona_calls.json")
    parser.add_argument('--instrumentation', action='store_true', default=False)
    parser.add_argument(
        '--mode',
        type=str,
        default="oracle",
        choices=["oracle", "limit"],
        help=
        "oracle means perfect dependency management, limit means no dependency is enforced thus all requests are sent in parallel"
    )
    args = parser.parse_args()

    dependency_adjacency = json.load(open(args.dag))
    persona_calls = json.load(open(args.trace))
    duplication = args.num_agents // 25

    if args.mode == "limit":
        bench_limit(persona_calls, duplication=duplication)
    elif args.mode == "oracle":
        bench_oracle(dependency_adjacency,
                     persona_calls,
                     rate_limit=args.rate_limit,
                     priority=args.priority,
                     duplication=duplication,
                     instrument=args.instrumentation)
