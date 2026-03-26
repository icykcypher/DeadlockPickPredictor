import subprocess
import time
import csv
import datetime
import re
import speedtest
import psutil


def get_ping():
    try:
        result = subprocess.run(
            ["ping", "-c", "4", "8.8.8.8"],
            capture_output=True,
            text=True,
            timeout=5
        )
        output = result.stdout

        times = re.findall(r'time=(\d+\.?\d*)', output)
        times = [float(t) for t in times]

        if len(times) == 0:
            return 999, 100, 0 

        avg_ping = sum(times) / len(times)
        # jitter
        diffs = [abs(times[i] - times[i-1]) for i in range(1, len(times))]
        jitter = sum(diffs) / len(diffs) if diffs else 0

        # packet loss
        loss_match = re.search(r'(\d+)% packet loss', output)
        loss = float(loss_match.group(1)) if loss_match else 0

        return avg_ping, loss, jitter
    except subprocess.TimeoutExpired:
        return 999, 100, 0

def get_wifi_signal():
    try:
        result = subprocess.run(["iw", "dev"], capture_output=True, text=True)
        output = result.stdout

        match = re.search(r'Interface\s+(\w+)', output)
        if not match:
            return 0

        interface = match.group(1)

        result = subprocess.run(["iw", "dev", interface, "link"], capture_output=True, text=True)
        output = result.stdout

        match = re.search(r'signal:\s*(-\d+)\s*dBm', output)
        if match:
            return int(match.group(1))
    except:
        pass
    return 0

def get_speed():
    try:
        st = speedtest.Speedtest()
        st.get_best_server()
        return st.download() / 1_000_000 
    except:
        return 0

def get_network_usage():
    net = psutil.net_io_counters()
    return net.bytes_sent + net.bytes_recv

with open("wifi_data3.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["timestamp", "rssi", "ping", "loss", "jitter", "speed", "network_usage", "hour", "label"])

    last_speed = 0
    last_speed_time = 0

    for i in range(2000):  
        timestamp = datetime.datetime.now()
        hour = timestamp.hour

        rssi = get_wifi_signal()
        ping, loss, jitter = get_ping()
        net_usage = get_network_usage()

        if time.time() - last_speed_time > 60:
            print("Measuring speed...")
            last_speed = get_speed()
            last_speed_time = time.time()

        speed = last_speed

        label = 1 if (ping > 120 or loss > 2 or speed < 10 or jitter > 30) else 0

        writer.writerow([timestamp, rssi, ping, loss, jitter, speed, net_usage, hour, label])

        status = "BAD" if label == 1 else "GOOD"
        print(f"[{status}] RSSI={rssi} dBm, ping={ping:.1f}ms, loss={loss}%, jitter={jitter:.1f}, speed={speed:.2f} Mbps, network_usage={net_usage}")

        time.sleep(3)