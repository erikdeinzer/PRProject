import os
import subprocess
import sys

CONFIGS_ROOT = "configs"
TRAIN_SCRIPT = "train.py"
sys.stdout.reconfigure(encoding='utf-8')

def find_config_files(root):
    config_files = []
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if filename.endswith(".py"):
                config_files.append(os.path.join(dirpath, filename))
    return config_files

def main():
    configs = find_config_files(CONFIGS_ROOT)
    print(f"Found {len(configs)} config files.")
    for config in configs:
        print(f"\nRunning config: {config}")
        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"
        result = subprocess.run(
            ["python", TRAIN_SCRIPT, "--config", config],
            text=True,
            env=env,              # Keep the UTF-8 fix
            stdout=sys.stdout,    # Send output directly to terminal
            stderr=sys.stderr
        )
        print(result.stdout)
        if result.returncode != 0:
            print(f"Error running {config}:\n{result.stderr}")

if __name__ == "__main__":
    main()