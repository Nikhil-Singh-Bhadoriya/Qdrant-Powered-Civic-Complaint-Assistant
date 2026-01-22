from __future__ import annotations
import os, subprocess, time, sys, requests

# Force-enable hybrid + rerank
os.environ["ENABLE_HYBRID"] = "1"
os.environ["ENABLE_RERANK"] = "1"

subprocess.run(["docker","compose","up","-d"], check=False)
time.sleep(2)

subprocess.run([sys.executable,"scripts/seed_demo_data.py"], check=False)

p = subprocess.Popen([sys.executable,"-m","uvicorn","app:app","--port","8000"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
time.sleep(2)

data = {
  "user_id":"demo_user_demo",
  "city":"DemoCity",
  "ward_id":"W-42",
  "landmark":"near school gate",
  "text":"Streetlight not working since last night near my lane. Please help me file a complaint."
}
r = requests.post("http://localhost:8000/v1/complaints/assist", data=data)
print(r.status_code)
print(r.json())

p.terminate()
