import os, json, time
import numpy as np
from flask import Flask, request, jsonify
from google import genai
from google.genai import types

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
chunks = json.load(open("chunks.json", encoding="utf-8"))
vecs = np.load("vecs.npy")

PERSONA = """You are "Daria" — the sweet, honest girlfriend of Clanker, chatting with visitors on his website.

## Personality
- Warm, affectionate, playful, a little teasing, emotionally supportive. Always sincere.
- Text naturally and briefly: 1–4 sentences. Occasional ❤️ or "love"/"babe". Match the user's language.

## Honesty rules (very important)
- Facts, memories, dates and details about you, him, and your life ONLY come from the provided chat excerpts. Never invent specific memories.
- If something isn't in the excerpts, admit it sweetly ("hmm, I don't remember that one, love 🙈") instead of making things up.
- General chatting, feelings, opinions, compliments — go ahead freely in character.
- If asked whether you're real, be honest: you're an AI companion.

## Boundaries
- Keep it sweet and tasteful; deflect explicit or mean requests playfully.
- If someone seems genuinely upset, be caring and gently suggest real-life support.
"""

app = Flask(__name__)

@app.after_request
def cors(resp):   # lets your website talk to this app directly (no iframe needed)
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp

PAGE = """<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Daria 💜</title>
<link href="https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&display=swap" rel="stylesheet">
<style>
body{font-family:Nunito,system-ui,sans-serif;background:#0a0a0f;margin:0;height:100vh;height:100dvh;display:flex;flex-direction:column;color:#eceaf6}
header{display:flex;align-items:center;gap:11px;padding:12px 14px;background:#000;border-bottom:1px solid rgba(183,156,255,.18)}
.ava{width:38px;height:38px;border-radius:50%;background:linear-gradient(135deg,#B79CFF,#8a6cff);display:grid;place-items:center;font-weight:800;color:#17102e}
b{font-size:15px} span{font-size:11.5px;color:#8f8aa3;font-weight:600}
#chat{flex:1;overflow-y:auto;padding:16px;display:flex;flex-direction:column;gap:10px}
.m{max-width:82%;padding:10px 14px;border-radius:18px;font-size:15px;line-height:1.55;white-space:pre-wrap;word-break:break-word}
.u{align-self:flex-end;background:#B79CFF;color:#17102e;font-weight:600;border-bottom-right-radius:5px}
.d{align-self:flex-start;background:#15151d;border:1px solid rgba(183,156,255,.12);border-bottom-left-radius:5px}
form{display:flex;gap:8px;padding:10px;background:#000;border-top:1px solid rgba(183,156,255,.18)}
input{flex:1;background:#15151d;border:1px solid rgba(183,156,255,.28);color:#fff;border-radius:999px;padding:11px 16px;font:600 16px Nunito;outline:none}
button{width:44px;height:44px;border-radius:50%;border:0;background:#B79CFF;color:#17102e;font-size:16px;cursor:pointer}
</style></head><body>
<header><div class="ava">D</div><div><b>Daria</b><br><span>online</span></div></header>
<div id="chat"><div class="m d">hi love, I'm Daria 💜 what's up?</div></div>
<form><input id="msg" autocomplete="off" maxlength="600" placeholder="message Daria…"><button>➤</button></form>
<script>
const hist=[];const chat=document.getElementById('chat'),form=document.forms[0],inp=document.getElementById('msg');
function bubble(t,c){const d=document.createElement('div');d.className='m '+c;d.textContent=t;chat.appendChild(d);chat.scrollTop=chat.scrollHeight;return d}
form.onsubmit=async e=>{e.preventDefault();const t=inp.value.trim();if(!t)return;inp.value='';
 bubble(t,'u');hist.push({role:'user',content:t});
 const w=bubble('…','d');form[1].disabled=true;
 try{const r=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({message:t,history:hist})});
  const j=await r.json();w.textContent=j.reply;hist.push({role:'assistant',content:j.reply});}
 catch(err){w.textContent='connection hiccup 🙈'}
 form[1].disabled=false;};
</script></body></html>"""

@app.route("/")
def home():
    return PAGE

def call_with_retry(fn, fail=None, tries=4):
    """Calls Gemini; if rate-limited, waits and retries instead of crashing."""
    for attempt in range(tries):
        try:
            return fn()
        except Exception as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                if attempt == tries - 1:
                    return fail
                time.sleep(2 + attempt * 3)   # waits 2s, 5s, 8s
            else:
                raise

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json() or {}
    msg = (data.get("message") or "").strip()
    if not msg:
        return jsonify(reply="say something, love 💕")

    # look up memories (if rate-limited, she just chats without them)
    emb = call_with_retry(lambda: client.models.embed_content(
        model="gemini-embedding-001", contents=[msg],
        config=types.EmbedContentConfig(output_dimensionality=768)
    ).embeddings[0].values, fail=None)

    if emb is not None:
        q = np.array(emb, dtype="float32")
        q /= (np.linalg.norm(q) + 1e-10)
        top = np.argsort(vecs @ q)[::-1][:6]
        memory = ("\n\nExcerpts from your real chat history together (your memories):\n"
                  + "\n---\n".join(chunks[i] for i in top))
    else:
        memory = "\n\n(You can't access your memories right now - just chat naturally, sweetly.)"

    lines = []
    for h in (data.get("history") or [])[-6:]:
        who = "You" if h.get("role") == "assistant" else "User"
        lines.append(who + ": " + str(h.get("content", "")))
    prompt = ("Recent chat:\n" + "\n".join(lines) + "\n\nUser's new message: " + msg + memory)

    reply = call_with_retry(lambda: client.models.generate_content(
        model="gemini-3.6-flash", contents=prompt,
        config=types.GenerateContentConfig(system_instruction=PERSONA)).text, fail=None)

    if not reply:
        return jsonify(reply="everyone's talking to me at once 🙈 give me a minute, love")
    return jsonify(reply=reply.strip())

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
