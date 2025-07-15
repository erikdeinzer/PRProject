import os
import subprocess
import sys
import time
import threading

CONFIGS_ROOT = "configs"
TRAIN_SCRIPT = "train.py"
MAX_PARALLEL = 4  # Adjust to your system

sys.stdout.reconfigure(encoding='utf-8')

def find_config_files(root):
    config_files = []
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if filename.endswith(".py"):
                config_files.append(os.path.join(dirpath, filename))
    return config_files

def stream_output(process, cfg):
    for line in process.stdout:
        print(f"[{os.path.basename(cfg)}] {line}", end="")

def main():
    configs = find_config_files(CONFIGS_ROOT)
    print(f"Found {len(configs)} config files.")

    processes = []
    for config in configs:
        while len(processes) >= MAX_PARALLEL:
            for p, cfg in processes:
                if p.poll() is not None:
                    print(f"✅ Finished: {cfg} (code {p.returncode})")
                    processes.remove((p, cfg))
            time.sleep(1)

        print(f"Launching: {config}")
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        p = subprocess.Popen(
            ["python", TRAIN_SCRIPT, "--config", config],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        threading.Thread(target=stream_output, args=(p, config), daemon=True).start()
        processes.append((p, config))

    # Wait for all to complete
    for p, cfg in processes:
        p.wait()
        print(f"Finished: {cfg} (code {p.returncode})")

if __name__ == "__main__":
    main()