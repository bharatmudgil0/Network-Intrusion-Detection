
import os
import glob
import subprocess
import time
from datetime import datetime

# =====================================================
# FOLDER CONFIGURATION
# =====================================================

CAPTURE_FOLDER = "live_capture/captures"
FLOW_FOLDER = "live_capture/flows"
PROCESSED_FOLDER = "live_capture/processed"

os.makedirs(FLOW_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Check for new PCAP files every 5 seconds
CHECK_INTERVAL = 5


# =====================================================
# PROCESS PCAP FILES
# =====================================================

def process_new_batches():

    pcap_files = glob.glob(
        os.path.join(CAPTURE_FOLDER, "live_*.pcap")
    )

    if not pcap_files:
        return

    for pcap_file in sorted(pcap_files):

        filename = os.path.basename(pcap_file)

        processed_marker = os.path.join(
            PROCESSED_FOLDER,
            filename + ".done"
        )

        # Skip already processed PCAP files
        if os.path.exists(processed_marker):
            continue

        flow_filename = (
            os.path.splitext(filename)[0] + ".csv"
        )

        flow_path = os.path.join(
            FLOW_FOLDER,
            flow_filename
        )

        print("\n[FLOW] Processing:", filename, flush=True)

        command = [
            "cicflowmeter",
            "-f", pcap_file,
            "-c", flow_path
        ]

        try:

            result = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                timeout=300
            )

            if os.path.exists(flow_path) and os.path.getsize(flow_path) > 0:

                print(
                    "[FLOW] Successfully extracted:",
                    flow_filename,
                    flush=True
                )

            else:

                print(
                    "[FLOW] No flow records generated:",
                    filename,
                    flush=True
                )

            # Mark this PCAP as processed after successful conversion
            with open(processed_marker, "w") as f:
                f.write(datetime.now().isoformat())

        except subprocess.TimeoutExpired:

            print(
                "[ERROR] Conversion timed out:",
                filename,
                flush=True
            )

        except subprocess.CalledProcessError as error:

            print(
                "[ERROR] Conversion failed:",
                filename,
                flush=True
            )

            print(error.stderr, flush=True)

        except Exception as error:

            print(
                "[ERROR]",
                error,
                flush=True
            )


# =====================================================
# CONTINUOUS PROCESSING LOOP
# =====================================================

if __name__ == "__main__":

    print("=" * 50)
    print("ML-NIDS FLOW PROCESSOR STARTED")
    print("Watching for new PCAP batches...")
    print("Press Ctrl+C to stop.")
    print("=" * 50)

    try:

        while True:

            process_new_batches()

            time.sleep(CHECK_INTERVAL)

    except KeyboardInterrupt:

        print("\n[FLOW] Processor stopped by user.")
