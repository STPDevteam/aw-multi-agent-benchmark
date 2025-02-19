To start the benchmark:

0. Install the dependency:
```
git clone git@github.com:AWE-Network/aw-engine.git

# note to install lfs, redis-server, pybind and hiredis library if not already
sudo apt-get update
sudo apt-get install redis
sudo apt-get install libhiredis-dev
sudo apt-get install git-lfs
sudo apt -y install python3-pybind11

# change to the repo directory
cd aw-engine
python -m pip install -e .

git lfs fetch
git lfs pull
```

1. Start the SGLang server and Redis server
```
# note to disable the radix cache, change the model, tp and dp setting as you need
python3 -m sglang.launch_server --model-path meta-llama/Meta-Llama-3-70B-Instruct --port 30000 --disable-radix-cache --tp 4

redis-server
```

2. Start the benchmark
```
python benchmark.py
# the benchmark result will be written into logs.jsonl
```
