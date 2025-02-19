import time
import json
import multiprocessing

from aw_engine import Simulator
from aw_engine.generative_agents_simple import TheVille, SimpleAgent
from aw_engine.utils import clear_all_db, dump_all_db, translate_traces, build_dependency_graph, find_critical_path
from aw_engine.generative_agents_simple.ville import ASSEST_PATH

SETTINGS = ["critical", "oracle", "async", "sync", "single-thread"]

if __name__ == '__main__':
    for s in SETTINGS:
        clear_all_db()
        base_step = 0
        target_step = 8640
        num_processes = num_agents = 25
        sim = Simulator(TheVille, base_step=base_step, env_args={"num_agents": num_agents})
        if s == "oracle" or s == "critical":
            movements = json.load(open(ASSEST_PATH + "movement_8640steps.json"))
            dependency_dag, dependency_reverse_dag = build_dependency_graph(movements, base_step, target_step)
            for key in dependency_dag:
                assert key in dependency_reverse_dag
                for blocking_key in dependency_dag[key]:
                    sim.env.db.sadd(f"oracle_dependency:{key}", blocking_key)
                for update_key in dependency_reverse_dag[key]:
                    sim.env.db.sadd(f"oracle_dependency_current:{key}", update_key)

            if s == "critical":
                traces = json.load(open(ASSEST_PATH + "traces_25agents_1day.json"))
                critical_path = find_critical_path(dependency_dag, traces)

                keys = sim.env.db.keys("recorded_calls:*")
                # only to keep requests that are in the critical path
                for key in keys:
                    if ":".join(key.split(":")[1:]) not in critical_path:
                        sim.env.db.delete(key)

        multiprocessing.freeze_support()
        print(f"Running benchmark for {s}")
        tic = time.time()
        sim.run(target_step, SimpleAgent, num_processes=num_processes, mode=s, priority=True)
        duration = time.time() - tic
        dump_all_db(f"benchmark_{s}_{num_agents}_{base_step}_{target_step}")
        duration_net = translate_traces(f"benchmark_{s}_{num_agents}_{base_step}_{target_step}/trace_db.json")
        print(
            f"Simulation for {s} from step {base_step} to {target_step} for {num_agents} agents took {duration:.2f}/{duration_net:.2f} seconds."
        )
