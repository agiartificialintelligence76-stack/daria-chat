import os, json
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

PAGE = """<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Daria</title><style>
body{font-family:system-ui,sans-serif;background:linear-gradient(160deg,#ffe3ec,#ffd6e7);margin:0;height:100vh;display:flex;flex-direction:column}
header{padding:14px;font-weight:600;color:#a4133c;background:#fff0f5;border-bottom:1px solid #ffc2d4}
#chat{flex:1;overflow-y:auto;padding:14px;display:flex;flex-direction:column;gap:8px}
.m{max-width:78%;padding:9px 13px;border-radius:16px;line-height:1.45;white-space:pre-wrap;word-wrap:break-word}
.u{align-self:flex-end;background:#ff5c8a;color:#fff;border-bottom-right-radius:4px}
.d{align-self:flex-start;background:#fff;color:#333;border-bottom-left-radius:4px;box-shadow:0 1px 3px rgba(0,0,0,.08)}
form{display:flex;gap:8px;padding:10px;background:#fff0f5;border-top:1px solid #ffc2d4}
input{flex:1;border:1px solid #ffc2d4;border-radius:20px;padding:10px 14px;outline:none;font-size:15px}
button{border:0;background:#ff5c8a;color:#fff;border-radius:20px;padding:10px 16px;font-size:15px;cursor:pointer}
</style></head><body>
<header>💕 Daria</header>
<div id="chat"><div class="m d">hi love, I'm Daria 💕 ask me anything</div></div>
<form><input id="msg" autocomplete="off" placeholder="type a message..."><button>➤</button></form>
<script>
const hist=[];const chat=document.getElementById('chat'),form=document.forms[0],inp=document.getElementById('msg');
function bubble(t,c){const d=document.createElement('div');d.className='m '+c;d.textContent=t;chat.appendChild(d);chat.scrollTop=chat.scrollHeight;return d}
form.onsubmit=async e=>{e.preventDefault();const t=inp.value.trim();if(!t)return;inp.value='';
 bubble(t,'u');hist.push({role:'user',content:t});
 const wait=bubble('...','d');form[1].disabled=true;
 try{const r=await fetch('/chat',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({message:t,history:hist})});
  const j=await r.json();wait.textContent=j.reply;hist.push({role:'assistant',content:j.reply});}
 catch(err){wait.textContent='connection hiccup, try again 🙈'}
 form[1].disabled=false;};
</script></body></html>"""

@app.route("/")
def home():
    return PAGE

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json() or {}
    msg = (data.get("message") or "").strip()
    if not msg:
        return jsonify(reply="say something, love 💕")
    q = np.array(client.models.embed_content(
        model="gemini-embedding-001", contents=[msg],
        config=types.EmbedContentConfig(output_dimensionality=768)
    ).embeddings[0].values, dtype="float32")
    q /= (np.linalg.norm(q) + 1e-10)
    top = np.argsort(vecs @ q)[::-1][:6]
    excerpts = "\n---\n".join(chunks[i] for i in top)
    lines = []
    for h in (data.get("history") or [])[-6:]:
        who = "You" if h.get("role") == "assistant" else "User"
        lines.append(who + ": " + str(h.get("content", "")))
    prompt = ("Recent chat:\n" + "\n".join(lines) + "\n\nUser's new message: " + msg +
              "\n\nExcerpts from your real chat history together (your memories):\n" + excerpts)
    resp = client.models.generate_content(
        model="gemini-3.6-flash", contents=prompt,
        config=types.GenerateContentConfig(system_instruction=PERSONA))
    return jsonify(reply=(resp.text or "hmm, my mind went blank 🙈").strip())

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
