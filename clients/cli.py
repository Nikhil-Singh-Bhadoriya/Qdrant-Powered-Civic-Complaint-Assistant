from __future__ import annotations
import argparse, requests

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000/v1/complaints/assist")
    ap.add_argument("--user_id", default="demo_user_cli")
    ap.add_argument("--city", default="DemoCity")
    ap.add_argument("--ward_id", default="W-42")
    ap.add_argument("--landmark", default="near school gate")
    ap.add_argument("--text", required=True)
    ap.add_argument("--photo", default=None)
    ap.add_argument("--screenshot", default=None)
    ap.add_argument("--audio", default=None)
    ap.add_argument("--api_key", default=None)
    args = ap.parse_args()

    files = {}
    if args.photo: files["issue_photo"] = open(args.photo, "rb")
    if args.screenshot: files["screenshot"] = open(args.screenshot, "rb")
    if args.audio: files["audio"] = open(args.audio, "rb")

    data = {"user_id": args.user_id, "city": args.city, "ward_id": args.ward_id, "landmark": args.landmark, "text": args.text}
    headers = {"X-API-Key": args.api_key} if args.api_key else {}
    r = requests.post(args.url, data=data, files=files if files else None, headers=headers)
    print(r.status_code)
    print(r.text)

if __name__ == "__main__":
    main()
