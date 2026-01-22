from __future__ import annotations
from PIL import Image, ImageDraw
from qdrant_client.http import models as qm

from core.qdrant_store import get_client, ensure_collections
from core.embeddings import embed_text, embed_image

CITY="DemoCity"
STATE="DemoState"
LANG="en"

def upsert(collection: str, points):
    client = get_client()
    ensure_collections(client)
    client.upsert(collection_name=collection, points=points)

def make_demo_image(label: str):
    im = Image.new("RGB",(256,256),(245,245,245))
    dr = ImageDraw.Draw(im)
    dr.rectangle([15,15,241,241], outline=(0,0,0), width=3)
    dr.text((20,120), label, fill=(0,0,0))
    return im

def main():
    client = get_client()
    ensure_collections(client)

    kb = [
        (1, "Pothole complaint procedure: Provide exact location, nearest landmark, date/time, and a clear photo. Municipal Roads handles potholes. SLA: 7 days. Channels: portal, app, helpline, email.",
         {"city":CITY,"state":STATE,"language":LANG,"department":"Municipal Roads","category":"Pothole","channel_type":["portal","app","helpline","email"],
          "required_fields":["location","landmark","photo","date_time"],"sla_days":7,"source":"official_portal","last_updated":"2026-01-01"}),
        (2, "Streetlight complaint: Provide location, landmark, date/time, pole number if available, plus photo. Electrical dept handles streetlights. SLA: 5 days. Channels: helpline, portal, email.",
         {"city":CITY,"state":STATE,"language":LANG,"department":"Electrical","category":"Streetlight","channel_type":["helpline","portal","email"],
          "required_fields":["location","landmark","photo","date_time","pole_number_optional"],"sla_days":5,"source":"official_portal","last_updated":"2026-01-01"}),
        (3, "Garbage not collected: Provide location, landmark, days missed, date/time, and photo. Sanitation dept handles garbage pickup. SLA: 2 days. Channels: app, helpline, portal.",
         {"city":CITY,"state":STATE,"language":LANG,"department":"Sanitation","category":"Garbage","channel_type":["app","helpline","portal"],
          "required_fields":["location","landmark","photo","date_time","days_missed"],"sla_days":2,"source":"official_portal","last_updated":"2026-01-01"}),
    ]
    kb_points = [qm.PointStruct(id=i, vector={"dense_text": embed_text(t)[0].tolist()}, payload={**p,"text":t}) for i,t,p in kb]
    upsert("civic_kb", kb_points)

    dir_rows = [
        (101, "Ward W-42, Zone Z-3: Roads issues handled by Municipal Roads. Helpline: 1800-ROADS. Email: roads@democity.gov.",
         {"city":CITY,"ward_id":"W-42","zone":"Z-3","department":"Municipal Roads","helpline":"1800-ROADS","email":"roads@democity.gov"}),
        (102, "Ward W-42, Zone Z-3: Streetlights handled by Electrical dept. Helpline: 1800-LIGHT. Email: electrical@democity.gov.",
         {"city":CITY,"ward_id":"W-42","zone":"Z-3","department":"Electrical","helpline":"1800-LIGHT","email":"electrical@democity.gov"}),
        (103, "Ward W-42, Zone Z-3: Garbage handled by Sanitation dept. Helpline: 1800-CLEAN. Email: sanitation@democity.gov.",
         {"city":CITY,"ward_id":"W-42","zone":"Z-3","department":"Sanitation","helpline":"1800-CLEAN","email":"sanitation@democity.gov"}),
    ]
    dir_points = [qm.PointStruct(id=i, vector={"dense_text": embed_text(t)[0].tolist()}, payload={**p,"text":t}) for i,t,p in dir_rows]
    upsert("jurisdiction_directory", dir_points)

    templates = [
        (201, "Pothole portal template (formal).",
         {"category":"Pothole","tone":"formal","channel_type":"portal","max_chars":"1000",
          "template":"Subject: Pothole Repair Required\n\nDear Municipal Roads Department,\n\nI would like to report a pothole at {location} near {landmark}. Observed on {date_time}. Please arrange inspection and repair at the earliest as it is a safety risk.\n\nAttachments: {attachments}\n\nRegards,\n{sender_name_optional}\n"}),
        (202, "Streetlight helpline template (concise).",
         {"category":"Streetlight","tone":"concise","channel_type":"helpline","max_chars":"400",
          "template":"Streetlight not working at {location}, near {landmark}. Observed: {date_time}. Pole no: {pole_number_optional}. Photo: {attachments}. Please resolve within SLA."}),
        (203, "Garbage app template (formal).",
         {"category":"Garbage","tone":"formal","channel_type":"app","max_chars":"800",
          "template":"Garbage has not been collected for {days_missed} days at {location} near {landmark}. Observed on {date_time}. Kindly arrange immediate pickup.\nAttachments: {attachments}"}),
    ]
    tpl_points = [qm.PointStruct(id=i, vector={"dense_text": embed_text(d+' '+p["template"])[0].tolist()}, payload={**p,"text":d}) for i,d,p in templates]
    upsert("complaint_templates", tpl_points)

    img_pothole = make_demo_image("POTHOLE")
    img_garbage = make_demo_image("GARBAGE")
    img_light = make_demo_image("STREETLIGHT")
    cases = [
        (301, "Case: Pothole near school. Resolution: barricaded within 1 day, patched in 3 days. Tip: add landmark + photo.",
         {"category":"Pothole","area_type":"school_zone"}, img_pothole),
        (302, "Case: Garbage missed pickup. Resolution: pickup within 1 day after helpline + photo. Tip: mention days missed.",
         {"category":"Garbage","area_type":"residential"}, img_garbage),
        (303, "Case: Streetlight off. Resolution: driver replaced in 2 days. Tip: include pole number if visible.",
         {"category":"Streetlight","area_type":"residential"}, img_light),
    ]
    case_points = []
    for i,t,p,img in cases:
        case_points.append(qm.PointStruct(id=i, vector={"dense_text": embed_text(t)[0].tolist(), "dense_image": embed_image(img).tolist()}, payload={**p,"text":t}))
    upsert("case_library", case_points)

    upsert("channel_status", [qm.PointStruct(id=401, vector={"dense_text": embed_text("Portal operational")[0].tolist()},
                                           payload={"city":CITY,"channel":"portal","status":"up","timestamp":"2026-01-21T00:00:00Z"})])

    print("Seeded demo data successfully.")

if __name__ == "__main__":
    main()
