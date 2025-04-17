import subprocess
import time
from config import NUM_CLIENTS


processes = []
for cid in range(NUM_CLIENTS):
    print(f"Launching client {cid}")
    p = subprocess.Popen(["python", "client.py", str(cid)])
    processes.append(p)
    time.sleep(1)  # stagger startups slightly

#wait for all clients to finish
for p in processes:
    p.wait()