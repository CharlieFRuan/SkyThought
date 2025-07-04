import subprocess
import time

TASK = "aime25"
TP_SIZE = 2
BATCH_SIZE = 32
MAX_NUM_SEQS = 64  # otherwise preemption occurs a lot on H100 TP=2
N = 16
# from Qwen3's recommended settings
SAMPLING_PARAMS = "temperature=0.6,top_p=0.95,top_k=20,min_p=0"

dtypes = ["float32", "float16", "bfloat16", "naive_fp8", "provided_fp8", "awq"]
sampling_strategies = ["greedy", f"passk_n{N}"]

for sampling_strategy in sampling_strategies:
    for dtype in dtypes:
        time.sleep(5)
        result_dir = f"results/{sampling_strategy}-{TASK}-{dtype}"
        model = "/home/ubuntu/models/Qwen3-8B"
        if dtype == "awq":
            model = "/home/ubuntu/models/Qwen3-8B-AWQ"
        elif dtype == "provided_fp8":
            model = "/home/ubuntu/models/Qwen3-8B-FP8"
        backend_args = f"tensor_parallel_size={TP_SIZE},max_num_seqs={MAX_NUM_SEQS}"
        if dtype == "naive_fp8":
            backend_args += ",quantization=fp8"
        elif dtype == "provided_fp8" or dtype == "awq":
            pass
        else:
            backend_args += f",dtype={dtype}"
        command = []
        if sampling_strategy == "greedy":
            command = [
                "skythought", "evaluate",
                "--model", model,
                "--task", TASK,
                "--backend", "vllm",
                "--backend-args", backend_args,
                "--batch-size", str(BATCH_SIZE),
                "--result-dir", result_dir
            ]
        else:
            command = [
                "skythought", "evaluate",
                "--model", model,
                "--task", TASK,
                "--backend", "vllm",
                "--backend-args", backend_args,
                "--sampling-params", SAMPLING_PARAMS,
                "--n", str(N),
                "--batch-size", str(BATCH_SIZE),
                "--result-dir", result_dir
            ]
        print(command)
        subprocess.run(command)
