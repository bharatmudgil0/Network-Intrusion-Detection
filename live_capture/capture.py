import os
import sys
from datetime import datetime

from scapy.all import (
    get_if_list,
    get_if_addr,
    sniff,
    wrpcap
)


# =====================================================
# 1. CONFIGURATION
# =====================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CAPTURE_DIR = os.path.join(
    BASE_DIR,
    "live_capture",
    "captures"
)

os.makedirs(CAPTURE_DIR, exist_ok=True)


# =====================================================
# 2. DISPLAY AVAILABLE NETWORK INTERFACES
# =====================================================

def list_interfaces():

    interfaces = get_if_list()

    print("\n========== AVAILABLE NETWORK INTERFACES ==========\n")

    for index, interface in enumerate(interfaces):

        try:
            ip_address = get_if_addr(interface)
        except Exception:
            ip_address = "Unavailable"

        print(f"{index}. Interface : {interface}")
        print(f"   IP Address : {ip_address}")
        print("----------------------------------------")

    return interfaces


# =====================================================
# 3. CAPTURE NETWORK PACKETS
# =====================================================

def capture_packets(interface, duration=30):

    print("\nStarting network packet capture...")

    print(f"Selected interface: {interface}")
    print(f"Capture duration: {duration} seconds")

    packets = sniff(
        iface=interface,
        timeout=duration,
        store=True
    )

    if not packets:

        print("\nNo packets captured.")
        return None

    filename = (
        "capture_"
        + datetime.now().strftime("%Y%m%d_%H%M%S")
        + ".pcap"
    )

    output_path = os.path.join(
        CAPTURE_DIR,
        filename
    )

    wrpcap(output_path, packets)

    print("\nCapture completed successfully.")

    print(f"Total packets: {len(packets)}")
    print(f"Saved file: {output_path}")

    return output_path


# =====================================================
# 4. MAIN FUNCTION
# =====================================================

if __name__ == "__main__":

    interfaces = list_interfaces()

    if not interfaces:
        print("No network interfaces found.")
        sys.exit(1)

    try:

        choice = int(
            input("\nEnter the interface number to monitor: ")
        )

        if choice < 0 or choice >= len(interfaces):
            raise ValueError("Invalid interface number.")

        selected_interface = interfaces[choice]

        capture_packets(
            interface=selected_interface,
            duration=30
        )

    except ValueError as error:

        print(f"Invalid selection: {error}")

    except PermissionError:

        print(
            "Permission denied. Run VS Code or the terminal "
            "with Administrator privileges."
        )

    except Exception as error:

        print(f"Capture error: {error}")