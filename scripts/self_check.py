from __future__ import annotations
import sys, json, time
import requests
import subprocess
from pathlib import Path

def main():
    # assumes API is running at localhost:8000
    url = "http://127.0.0.1:8000/v1/complaints/assist"
    data = {"user_id":"selftest","city":"DemoCity","ward_id":"W-42","landmark":"near school","text":"pothole on road", "intent":"new", "auto_submit":"true"}
    r = requests.post(url, data=data)
    assert r.status_code == 200, r.text
    js = r.json()
    assert "recommended_action" in js or js.get("need_more_info"), js
    if not js.get("need_more_info"):
        assert "ticket_id" in js, js
        tid = js["ticket_id"]
        r2 = requests.post(url, data={"user_id":"selftest","city":"DemoCity","ward_id":"W-42","landmark":"","text":"", "intent":"track", "ticket_id":tid})
        assert r2.status_code == 200, r2.text
    print("Self-check OK")

if __name__ == "__main__":
    main()
