
import subprocess
import sys
import time
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

processes = []

scripts = [
    "live_capture/continuous_capture.py",
    "live_capture/process_batches.py",
    "live_capture/predict_batches.py"
]

try:
    for script in scripts:

        path = os.path.join(ROOT, script)

        print(f"\nStarting {script}...")

        process = subprocess.Popen(
            [sys.executable, path],
            cwd=ROOT
        )

        processes.append(process)

        time.sleep(2)

    print("\nAll background workers started.")
    print("Starting Streamlit...\n")

    streamlit = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "app.py"
        ],
        cwd=ROOT
    )

    processes.append(streamlit)

    while True:

        for process in processes:

            if process.poll() is not None:

                raise RuntimeError(
                    f"A process stopped unexpectedly: {process.args}"
                )

        time.sleep(2)

except KeyboardInterrupt:

    print("\nStopping ML-NIDS...")

except Exception as error:

    print(f"\nError: {error}")

finally:

    for process in reversed(processes):

        if process.poll() is None:

            process.terminate()

    for process in processes:

        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()

    print("All ML-NIDS processes stopped.")
