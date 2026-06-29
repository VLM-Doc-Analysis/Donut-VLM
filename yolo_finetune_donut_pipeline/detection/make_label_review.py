# 값 라벨 검수 도구 생성 — 서버 없이 브라우저로 여는 자체완결 HTML
# 사용: conda activate donut_vml; python yolo_finetune_donut_pipeline/detection/make_label_review.py
# 입력: data/elements_hidpi/{values_hidpi.jsonl, images/}  (prefill_values.py 산출)
# 출력: data/elements_hidpi/review.html  ← 브라우저로 열어 검수 → "Export" 로 reviewed.jsonl 다운로드
#        → ingest_reviewed.py 로 라벨에 반영.
import os, json, base64, io
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
HID = os.path.join(REPO, "data/elements_hidpi")
CLASSES = ["Dimension", "GD&T_FCF", "Datum", "Surface_Roughness", "Section", "Hole_Callout"]
MAXSIDE = 760  # 임베드 이미지 최대 변(가독 + 용량 균형)

rows = []
for ln in open(os.path.join(HID, "values_hidpi.jsonl"), encoding="utf-8"):
    ln = ln.strip()
    if not ln: continue
    d = json.loads(ln); ip = os.path.join(HID, "images", d["crop"])
    if not os.path.exists(ip): continue
    im = Image.open(ip).convert("RGB")
    if max(im.size) > MAXSIDE:
        r = MAXSIDE / max(im.size); im = im.resize((int(im.width*r), int(im.height*r)))
    buf = io.BytesIO(); im.save(buf, "PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    rows.append({"crop": d["crop"], "cls": d.get("class", "value"), "val": d.get("value", ""),
                 "img": "data:image/png;base64," + b64})

TEMPLATE = r"""<!doctype html><html lang=ko><head><meta charset=utf-8>
<title>Element 값 라벨 검수</title><style>
body{font-family:system-ui,sans-serif;margin:0;background:#f4f4f4}
#bar{position:sticky;top:0;background:#222;color:#fff;padding:8px 12px;display:flex;gap:10px;align-items:center;flex-wrap:wrap;z-index:9}
#bar b{font-size:15px} .flt{background:#444;color:#fff;border:0;padding:4px 9px;border-radius:5px;cursor:pointer}
.flt.on{background:#1a73e8} #exp{background:#0a0;color:#fff;border:0;padding:6px 12px;border-radius:5px;cursor:pointer;font-weight:700}
#prog{margin-left:auto}
.row{display:flex;gap:10px;align-items:center;background:#fff;margin:6px 10px;padding:8px;border-radius:6px;border-left:6px solid #ccc}
.row.ok{border-left-color:#0a0;background:#f3fff3} .row.bad{border-left-color:#d22;background:#fff3f3;opacity:.6}
.row img{height:64px;background:#fafafa;border:1px solid #eee;transition:transform .1s} .row img:hover{transform:scale(2.4);transform-origin:left center;position:relative;z-index:5}
.row select{padding:4px} .row input.v{flex:1;padding:6px 8px;font-size:16px;font-family:monospace}
.row .id{font-size:11px;color:#999;width:230px;overflow:hidden;white-space:nowrap}
.bbtn{background:#d22;color:#fff;border:0;padding:5px 9px;border-radius:5px;cursor:pointer}
.row.bad .bbtn{background:#555}
</style></head><body>
<div id=bar>
 <b>Element 값 검수</b>
 <button class=flt data-c=ALL onclick=filt(this)>ALL</button>
 <span id=clsbtns></span>
 <button class=flt data-c=PENDING onclick=filt(this)>미검수만</button>
 <span id=prog></span>
 <button id=exp onclick=exportJSONL()>Export reviewed.jsonl</button>
</div>
<div id=list></div>
<script>
const CLASSES=/*CLASSES*/; const DATA=/*DATA*/;
const KEY="elemReview_v1";
let st=JSON.parse(localStorage.getItem(KEY)||"{}");  // crop -> {val,cls,status}
let curFilter="ALL";
function save(){localStorage.setItem(KEY,JSON.stringify(st))}
function rec(c){ if(!st[c]){const d=DATA.find(x=>x.crop==c); st[c]={val:d.val,cls:d.cls,status:"pending"}} return st[c]}
function setOk(c,v,cl){let r=rec(c); r.val=v; r.cls=cl; r.status="ok"; save(); paint()}
function toggleBad(c){let r=rec(c); r.status=(r.status=="bad")?"pending":"bad"; save(); paint()}
function prog(){let ok=0,bad=0; DATA.forEach(d=>{let s=st[d.crop]?st[d.crop].status:"pending"; if(s=="ok")ok++; if(s=="bad")bad++});
  document.getElementById("prog").textContent=`검수 ${ok+bad}/${DATA.length}  (ok ${ok} / bad ${bad})`}
function clsBtns(){let h=""; CLASSES.forEach(c=>h+=`<button class=flt data-c="${c}" onclick=filt(this)>${c}</button>`); document.getElementById("clsbtns").innerHTML=h}
function filt(btn){curFilter=btn.dataset.c; document.querySelectorAll(".flt").forEach(b=>b.classList.toggle("on",b.dataset.c==curFilter)); render()}
function render(){
 let html="";
 DATA.forEach((d,i)=>{
   let r=st[d.crop]||{val:d.val,cls:d.cls,status:"pending"};
   if(curFilter!="ALL"){ if(curFilter=="PENDING"){if(r.status!="pending")return} else if(r.cls!=curFilter)return }
   let opt=CLASSES.map(c=>`<option ${c==r.cls?"selected":""}>${c}</option>`).join("");
   html+=`<div class="row ${r.status}" id="row${i}">
     <span class=id>${d.crop}</span>
     <img src="${d.img}">
     <select onchange="setOk('${d.crop}',document.getElementById('vi${i}').value,this.value)">${opt}</select>
     <input class=v id="vi${i}" value="${(r.val||"").replace(/"/g,'&quot;')}"
        onkeydown="if(event.key=='Enter'){setOk('${d.crop}',this.value,document.getElementById('row${i}').querySelector('select').value);let n=document.getElementById('vi${i+1}');if(n)n.focus()}"
        onblur="setOk('${d.crop}',this.value,document.getElementById('row${i}').querySelector('select').value)">
     <button class=bbtn onclick="toggleBad('${d.crop}')">bad</button>
   </div>`;
 });
 document.getElementById("list").innerHTML=html; prog();
}
function paint(){ DATA.forEach((d,i)=>{let el=document.getElementById("row"+i); if(el){let r=st[d.crop]; el.className="row "+(r?r.status:"pending")}}); prog()}
function exportJSONL(){
 let lines=DATA.map(d=>{let r=st[d.crop]||{val:d.val,cls:d.cls,status:"pending"}; return JSON.stringify({crop:d.crop,class:r.cls,value:r.val,status:r.status})});
 let blob=new Blob([lines.join("\n")],{type:"application/x-ndjson"});
 let a=document.createElement("a"); a.href=URL.createObjectURL(blob); a.download="reviewed.jsonl"; a.click();
}
clsBtns(); document.querySelector('.flt[data-c=ALL]').classList.add('on'); render();
</script></body></html>"""

html = (TEMPLATE
        .replace("/*CLASSES*/", json.dumps(CLASSES, ensure_ascii=False))
        .replace("/*DATA*/", json.dumps(rows, ensure_ascii=False)))
out = os.path.join(HID, "review.html")
open(out, "w", encoding="utf-8").write(html)
print(f"검수 도구 생성: {out}  ({len(rows)} 크롭, {os.path.getsize(out)//1024} KB)")
print("브라우저로 열어 검수 → 'Export reviewed.jsonl' → ingest_reviewed.py 로 반영")
