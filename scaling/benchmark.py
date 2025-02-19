import os
import time
import json
import multiprocessing
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from aw_engine import Simulator
from aw_engine.generative_agents_simple import TheVille, SimpleAgent
from aw_engine.utils import clear_all_db, dump_all_db, translate_traces, build_dependency_graph, find_critical_path
from aw_engine.generative_agents_simple.ville import ASSEST_PATH
from aw_engine.generative_agents_simple.agent_functions import common_llm_call

INTERVAL = ["quiet", "busy"]
SETTINGS = ["no-dependency", "critical", "oracle", "async", "sync", "single-thread"]
NUM_WORLDS = [4, 1, 20, 100]
MAZE_HEIGHT = 100


def concat_traces(num_worlds, status):
    new_traces = {}
    new_movements = {}
    for i in range(num_worlds):
        for step, personas in json.load(open(f"excerpt_traces/{status}/{i}_trace.json")).items():
            if step not in new_traces:
                new_traces[step] = personas
            else:
                for persona, funcs in personas.items():
                    persona = f"{persona}_{i}"
                    assert persona not in new_traces[step]
                    new_traces[step][persona] = funcs
        for persona, movements in json.load(open(f"excerpt_traces/{status}/{i}_movement.json")).items():
            if persona not in new_movements:
                new_movements[persona] = movements
            else:
                persona = f"{persona}_{i}"
                movements = [(x, y + i * MAZE_HEIGHT) for x, y in movements]
                assert persona not in new_movements
                new_movements[persona] = movements
    return new_traces, new_movements


def bench_no_dependency(traces):
    calls = [call for v in traces.values() for call_list in v.values() for call in call_list]
    progress_bar = tqdm(total=len(calls))

    def worker(c):
        common_llm_call("", 0, c["prompt"], c["max_tokens"], c["stop"], c["ignore_eos"])
        progress_bar.update(1)

    start_time = time.time()

    with ThreadPoolExecutor(max_workers=os.cpu_count() * 8) as executor:
        futures = [executor.submit(worker, c) for c in calls]

        for future in as_completed(futures):
            future.result()

    print(f"Time taken: {time.time() - start_time}")


if __name__ == '__main__':
    for d in NUM_WORLDS:
        for i in INTERVAL:
            for s in SETTINGS:
                clear_all_db()
                if i == "busy":
                    base_step = 4320
                    target_step = 4680
                else:
                    base_step = 2160
                    target_step = 2520
                num_agents = 25 * d
                num_processes = min(os.cpu_count(), num_agents)
                traces, movements = concat_traces(d, i)

                if s != "no-dependency":
                    sim = Simulator(TheVille,
                                    base_step=base_step,
                                    env_args={
                                        "num_agents": num_agents,
                                        "cache": False,
                                        "recorded_movement": movements,
                                        "recorded_traces": traces
                                    })

                if s == "oracle" or s == "critical":
                    dependency_dag, dependency_reverse_dag = build_dependency_graph(movements, base_step, target_step)
                    for key in dependency_dag:
                        assert key in dependency_reverse_dag
                        for blocking_key in dependency_dag[key]:
                            sim.env.db.sadd(f"oracle_dependency:{key}", blocking_key)
                        for update_key in dependency_reverse_dag[key]:
                            sim.env.db.sadd(f"oracle_dependency_current:{key}", update_key)

                    if s == "critical":
                        critical_path = find_critical_path(dependency_dag, traces)
                        keys = sim.env.db.keys("recorded_calls:*")
                        # only to keep requests that are in the critical path
                        # print(critical_path)
                        for key in keys:
                            if ":".join(key.split(":")[1:]) not in critical_path:
                                sim.env.db.delete(key)

                multiprocessing.freeze_support()
                print(f"Running benchmark for {s} {i} with {num_agents} agents.")

                tic = time.time()
                if s == "no-dependency":
                    bench_no_dependency(traces)
                else:
                    sim.run(target_step, SimpleAgent, num_processes=num_processes, mode=s, priority=True)
                duration = time.time() - tic

                log_dir = f"benchmark_{s}_{i}_{num_agents}_{base_step}_{target_step}"
                dump_all_db(log_dir)
                duration_net = translate_traces(f"{log_dir}/trace_db.json")
                print(
                    f"Simulation for {s} {i} from step {base_step} to {target_step} for {num_agents} agents took {duration:.2f}/{duration_net:.2f} seconds."
                )

                log_json = {
                    "setting": s,
                    "interval": i,
                    "num_agents": num_agents,
                    "base_step": base_step,
                    "target_step": target_step,
                    "duration": duration,
                    "duration_net": duration_net
                }
                with open('logs.jsonl', 'a') as f:
                    f.write(json.dumps(log_json) + '\n')
