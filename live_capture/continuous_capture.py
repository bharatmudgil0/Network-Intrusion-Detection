
import os
import time
import threading
from datetime import datetime

from scapy.all import sniff, wrpcap

# Configuration
INTERFACE = r"\Device\NPF_{9855F949-2775-49C7-B5CC-8B966E8C69B7}"
CAPTURE_FOLDER = "live_capture/captures"
BATCH_DURATION = 30

os.makedirs(CAPTURE_FOLDER, exist_ok=True)

running = True
packets = []
lock = threading.Lock()


def packet_callback(packet):
    with lock:
        packets.append(packet)


def save_batch():
    global packets

    with lock:
        if not packets:
            return

        batch = packets.copy()
        packets.clear()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    filename = os.path.join(
        CAPTURE_FOLDER,
        f"live_{timestamp}.pcap"
    )

    wrpcap(filename, batch)

    print(
        f"[CAPTURE] Saved {len(batch)} packets: {filename}",
        flush=True
    )


def capture_packets():
    print("[CAPTURE] Starting packet capture...", flush=True)

    while running:
        try:
            sniff(
                iface=INTERFACE,
                prn=packet_callback,
                store=False,
                timeout=BATCH_DURATION
            )

            save_batch()

        except Exception as error:
            print("[ERROR]", error, flush=True)
            time.sleep(2)


if __name__ == "__main__":

    capture_thread = threading.Thread(
        target=capture_packets,
        daemon=True
    )

    capture_thread.start()

    print("[CAPTURE] Running. Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        running = False
        print("\n[CAPTURE] Stopping...")

        capture_thread.join(timeout=5)
        save_batch()

        print("[CAPTURE] Stopped.")
