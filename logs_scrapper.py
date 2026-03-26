import csv
import re
from datetime import datetime

input_file = "wifi_logs.txt"
output_file = "wifi_data_recovered2.csv"

with open(input_file, "r") as f:
    lines = f.readlines()

with open(output_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["timestamp", "rssi", "ping", "loss", "jitter", "speed", "network_usage", "hour", "label"])

    for line in lines:
        line = line.strip()
        if not line:
            continue

        status_match = re.match(r'\[(GOOD|BAD)\]', line)
        if not status_match:
            continue
        status = status_match.group(1)
        label = 0 if status == "GOOD" else 1

        rssi = re.search(r'RSSI=(-?\d+)', line)
        ping = re.search(r'ping=([\d.]+)', line)
        loss = re.search(r'loss=([\d.]+)', line)
        jitter = re.search(r'jitter=([\d.]+)', line)
        speed = re.search(r'speed=([\d.]+)', line)
        net_usage = re.search(r'network_usage=(\d+)', line)

        if all([rssi, ping, loss, jitter, speed, net_usage]):
            timestamp = datetime.now()
            hour = timestamp.hour
            writer.writerow([
                timestamp,
                int(rssi.group(1)),
                float(ping.group(1)),
                float(loss.group(1)),
                float(jitter.group(1)),
                float(speed.group(1)),
                int(net_usage.group(1)),
                hour,
                label
            ])