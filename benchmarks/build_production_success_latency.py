#!/usr/bin/env python3
"""PURE process_image latency (successfully finished image) by pairing the text
markers 'Starting function process_image (PID: N)' <-> 'Finishing function
process_image (PID: N)' by (container, PID). The camera comes from the
preceding line 'Image processor initialized for instrument_name'.
A Starting followed by another Starting on the same PID with no Finishing =
FAILURE (did not finish successfully).
Read-only. Usage: <days> [<gpu_containers_only 0/1>]"""
import sys, json, re, urllib.request, statistics as st
from collections import defaultdict
import os, base64
ES = os.environ.get("ES_URL", "http://10.0.210.30:9200")
# Credentials via environment: ES_USER/ES_PASS. Never in the repository.
AUTH = "Basic " + base64.b64encode(
    f'{os.environ["ES_USER"]}:{os.environ["ES_PASS"]}'.encode()).decode()
def q(body, path="/logstash-*/_search"):
    r=urllib.request.Request(ES+path, data=json.dumps(body).encode(),
        headers={"Content-Type":"application/json","Authorization":AUTH})
    return json.load(urllib.request.urlopen(r, timeout=180))
days=sys.argv[1] if len(sys.argv)>1 else "1"
# GPU container -> model, from production exceptions
a=q({"size":0,"query":{"bool":{"filter":[
      {"term":{"extra.function_name.keyword":"process_image"}},
      {"term":{"extra.environment.keyword":"production"}},
      {"range":{"@timestamp":{"gte":"now-120d"}}}]}},
     "aggs":{"c":{"terms":{"field":"logsource.keyword","size":200},
        "aggs":{"g":{"terms":{"field":"extra.system_info.gpu.name.keyword","size":3}}}}}})
SH={"NVIDIA A100-SXM4-80GB":"A100","NVIDIA L40S":"L40S","NVIDIA H100 PCIe":"H100"}
c2gpu={b["key"]:SH.get(b["g"]["buckets"][0]["key"],b["g"]["buckets"][0]["key"]) for b in a["aggregations"]["c"]["buckets"] if b["g"]["buckets"]}
gpuset=set(c2gpu)
# scroll the 3 markers
body={"size":4000,"_source":["message","logsource","@timestamp","extra.process_name"],
   "sort":[{"@timestamp":{"order":"asc"}}],
   "query":{"bool":{"filter":[
     {"range":{"@timestamp":{"gte":f"now-{days}d"}}},
     {"bool":{"should":[
        {"match_phrase":{"message":"Starting function process_image"}},
        {"match_phrase":{"message":"Finishing function process_image"}},
        {"match_phrase":{"message":"Image processor initialized"}},
        {"match_phrase":{"message":"Processing image:"}}],
        "minimum_should_match":1}}]}}}
res=q({**body}, "/logstash-*/_search?scroll=3m")
sid=res.get("_scroll_id"); docs=[]
def take(hits):
    for h in hits:
        s=h["_source"]; ls=s.get("logsource")
        if ls not in gpuset: continue
        docs.append((s.get("@timestamp"), ls, s.get("message",""), (s.get("extra") or {}).get("process_name","")))
take(res["hits"]["hits"])
while sid:
    res=q({"scroll":"3m","scroll_id":sid}, "/_search/scroll")
    hh=res["hits"]["hits"]
    if not hh: break
    take(hh); sid=res.get("_scroll_id")
    if len(docs)>500000: break
docs.sort(key=lambda x:(x[1], x[0]))   # by container then time
rxpid=re.compile(r"function process_image \(PID: (\d+)\)")
rxinst=re.compile(r"'instrument_name':\s*'([^']+)'")
rxproc=re.compile(r"Processing image:\s*(\S+)")
CAMMP={"QHY411":151,"QHY600":61,"iKon936":4}
def fam(c):
    for k in CAMMP:
        if c.startswith(k): return k
    if c.startswith("cam") or "ATLAS" in c: return "ATLAS-cam"
    return c
def ident_from_path(p):
    b=p.split("/")[-1]
    if "ATLAS" in p or b.startswith("ATLAS"): return "ATLAS-cam"
    parts=b.split("_")
    return parts[1] if len(parts)>1 else parts[0]
# state per (container,pid)
open_start={}          # (ls,pid)-> (ts, instrument)
last_inst=defaultdict(str)  # (ls, process_name) -> instrument
by=defaultdict(list); fails=defaultdict(int); paired=0
def parse_ts(t):
    from datetime import datetime
    return datetime.strptime(t[:23],"%Y-%m-%dT%H:%M:%S.%f").timestamp()
for ts,ls,m,pn in docs:
    if "Image processor initialized" in m:
        im=rxinst.search(m)
        if im: last_inst[(ls,pn)]=im.group(1)
        continue
    if m.startswith("Processing image:"):
        pm2=rxproc.search(m)
        if pm2: last_inst[(ls,pn)]=ident_from_path(pm2.group(1))
        continue
    pm=rxpid.search(m)
    if not pm: continue
    pid=pm.group(1); key=(ls,pid)
    if m.startswith("Starting function process_image"):
        if key in open_start:  # previous start had no finish -> failed
            fails[c2gpu.get(ls,"?")]+=1
        open_start[key]=(ts, last_inst.get((ls,pn),"?"))
    elif m.startswith("Finishing function process_image"):
        if key in open_start:
            ts0,inst=open_start.pop(key)
            dur=parse_ts(ts)-parse_ts(ts0)
            if 0<=dur<3000:
                by[(fam(inst), c2gpu.get(ls,"?"))].append(dur); paired+=1
# leftover open starts (no finish yet at window end) are not counted as fail
def pct(v,p):
    v=sorted(v); return v[min(len(v)-1,int(p/100*len(v)))]
print(f"# process_image PURO (Starting->Finishing) now-{days}d  emparejados={paired}")
print(f"# fam camara->Mpx {CAMMP}; iKon936=4.2MP; ATLAS-cam=survey")
print("camara_fam,Mpx,gpu,n,med_s,p05,p25,p75,p95,min,max")
for (f,g),v in sorted(by.items(),key=lambda x:(x[0][0],x[0][1])):
    if len(v)<3: continue
    print(f"{f},{CAMMP.get(f,'')},{g},{len(v)},{st.median(v):.1f},{pct(v,5):.1f},{pct(v,25):.1f},{pct(v,75):.1f},{pct(v,95):.1f},{min(v):.1f},{max(v):.1f}")
print("# fallos_detectados(Starting sin Finishing) por gpu:", dict(fails))
