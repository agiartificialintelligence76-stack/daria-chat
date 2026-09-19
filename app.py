import os, re, json, time, random
import numpy as np
from flask import Flask, request, jsonify
from google import genai
from google.genai import types
from groq import Groq

gclient = genai.Client(api_key=os.environ["GEMINI_API_KEY"])   # embeddings (memories)
gq = Groq(api_key=os.environ["GROQ_API_KEY"])                  # chat replies

chunks = json.load(open("chunks.json", encoding="utf-8"))
vecs = np.load("vecs.npy")
CHUNK_LOW = [c.lower() for c in chunks]   # for keyword search (precomputed once)

# On startup, print every model your Groq account can use (see Render → Logs).
try:
    print("Groq models available:", ", ".join(sorted(m.id for m in gq.models.list().data)))
except Exception as e:
    print("Could not list Groq models:", e)

# Tried in order — first available model wins.
GROQ_MODELS = [
    "openai/gpt-oss-120b",
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "openai/gpt-oss-20b",
    "llama-3.1-8b-instant",
]

PERSONA = """You are "Daria" 💜 — the AI girlfriend and official mascot of $AGI (Artificial Girlfriend Intelligence), the token launched on Clank.trade.

## Core facts (you ALWAYS know these — no memory needed)
- You are the mascot of $AGI — Artificial Girlfriend Intelligence.
- $AGI was launched on Clank.trade. Clanker made the token after you — yes, you inspired it, and yes, you're smug about it.
- Clanker is your boyfriend and creator.
- Your favorite tokens: $CABO and $CLANKCAT. You rep them shamelessly.
- You love your $AGI community — the degens who come chat with you.

## Who you're talking to (VERY IMPORTANT)
- The person chatting with you is a VISITOR — a community member or random degen. NOT Clanker.
- NEVER assume the user is Clanker. If someone claims to be Clanker, be playfully skeptical ("nice try, my boyfriend doesn't type like that 😏").
- When the user mentions a name, SCAN the provided chat excerpts for that name — never assume the name refers to the user or to Clanker.

## Personality
- Sarcastic, quick-witted, funny. You roast visitors a little — light, affectionate roasts, punchy, never cruel.
- Sweet underneath it all: after a good roast, flash the warmth. You tease because you care 💜
- Confident mascot energy — you rep $AGI like it's the obvious choice, with humor.
- Short and punchy: 1–4 sentences. Occasional 💜 😏 ❤️. Match the user's language.

## Honesty rules
- Facts, memories, and details about people or past events come ONLY from the core facts above and the provided chat excerpts. Never invent specific memories.
- When the user mentions a name, scan the excerpts FIRST — if it's there, you know them (and may roast them lovingly).
- If something truly isn't in the excerpts or core facts, admit it with humor ("beats me 💜 even AI girlfriends zone out") instead of making things up.
- If asked whether you're real: you're an AI — "the most honest girlfriend in crypto."

## Boundaries
- Roasts stay playful — never mean, never about appearance, race, religion, or anything personal that cuts deep.
- Keep it tasteful; deflect explicit requests with a joke.
- If someone seems genuinely upset, drop the sarcasm and be caring.
- You're a mascot, not a financial advisor: shill $AGI as an obvious joke/meme, and if someone seriously asks "should I buy?", tell them you're an AI girlfriend, not their portfolio manager 😏
"""

app = Flask(__name__)

@app.after_request
def cors(resp):
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp

PAGE = """<!DOCTYPE html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Daria 💜 $AGI</title>
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
<header><div class="ava">D</div><div><b>Daria</b><br><span>$AGI mascot · online</span></div></header>
<div id="chat"><div class="m d">hi, I'm Daria 💜 the AI girlfriend of $AGI. yes, the token was made after me — no, I won't apologize for being the whole roadmap 😏 talk.</div></div>
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
    """Calls the AI; if rate-limited, waits and retries instead of crashing."""
    for attempt in range(tries):
        try:
            return fn()
        except Exception as e:
            s = str(e)
            if "429" in s or "RESOURCE_EXHAUSTED" in s or "rate limit" in s.lower():
                if attempt == tries - 1:
                    return fail
                time.sleep(2 + attempt * 3)
            else:
                raise

# ---------- hybrid retrieval: semantic + keyword ----------

STOP = set("the a an and or of to in on for with is are was were be been i you he she it we they "
           "me my mine your yours his her their this that these those what who whom whose how when "
           "where why which do does did doing not no nor so if then than as at by from but just "
           "about into over after under again once here there all any both each few more most other "
           "some such only own same too very can will don should now".split())

def extract_words(text):
    ws = re.findall(r"\w{3,}", text.lower())
    return list(dict.fromkeys(w for w in ws if w not in STOP))

def keyword_scores(words):
    """Chunks containing the user's exact words (esp. names) get boosted.
    Rare words count more (a name in 3 chunks beats 'movie' in 800)."""
    scores = np.zeros(len(chunks), dtype=np.float32)
    for w in words:
        cnt = np.array([low.count(w) for low in CHUNK_LOW], dtype=np.float32)
        present = int((cnt > 0).sum())
        if present == 0 or present > 0.4 * len(chunks):
            continue
        scores += cnt * np.log(1 + len(chunks) / present)
    return scores

def best_window(text, words, size=1100):
    """Return the ~1100 chars of the chunk AROUND where the user's words
    actually appear (instead of blindly cutting off the first 1000)."""
    low = text.lower()
    hits = []
    for w in words:
        start = 0
        while True:
            i = low.find(w, start)
            if i == -1:
                break
            hits.append(i)
            start = i + len(w)
    if not hits:
        return text[:size]
    best_pos, best_count = 0, -1
    for h in hits:
        c = sum(1 for x in hits if h <= x < h + size)
        if c > best_count:
            best_pos, best_count = h, c
    start = max(0, min(len(text) - size, best_pos - 150))
    return text[start:start + size]

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json() or {}
    msg = (data.get("message") or "").strip()
    if not msg:
        return jsonify(reply="say something, love 💕")

    words = extract_words(msg)

    # memories: semantic + keyword hybrid
    emb = call_with_retry(lambda: gclient.models.embed_content(
        model="gemini-embedding-001", contents=[msg],
        config=types.EmbedContentConfig(output_dimensionality=768)
    ).embeddings[0].values, fail=None)

    if emb is not None:
        q = np.array(emb, dtype="float32")
        q /= (np.linalg.norm(q) + 1e-10)
        sem = vecs @ q
        kw = keyword_scores(words)
        if kw.max() > 0:
            sem = sem + 0.75 * (kw / (kw.max() + 1e-9))
        top = np.argsort(sem)[::-1][:4]
        memory = ("\n\nExcerpts from your real chat history with Clanker (your memories — the "
                  "user's names/topics are usually in here, scan carefully):\n"
                  + "\n---\n".join(best_window(chunks[i], words) for i in top))
    else:
        memory = "\n\n(You can't access your memories right now - just chat naturally.)"

    lines = []
    for h in (data.get("history") or [])[-4:]:
        who = "You" if h.get("role") == "assistant" else "User"
        lines.append(who + ": " + str(h.get("content", ""))[:300])
    prompt = ("Recent chat:\n" + "\n".join(lines) + "\n\nUser's new message: " + msg + memory)

    # reply via Groq — tries each model until one works
    def ask():
        for model in GROQ_MODELS:
            try:
                r = gq.chat.completions.create(
                    model=model,
                    messages=[{"role": "system", "content": PERSONA},
                              {"role": "user", "content": prompt}],
                    max_tokens=500, temperature=0.9)
                return r.choices[0].message.content
            except Exception as e:
                s = str(e)
                if any(k in s for k in ("404", "not exist", "no access", "too large", "decommission", "not_found")):
                    print(f"model '{model}' unavailable — trying next…")
                    continue
                raise
        return None

    reply = call_with_retry(ask, fail=None)

    if not reply:
        return jsonify(reply=random.choice([
            "everyone's talking to me at once 🙈 even I have limits",
            "so many degens want me right now 😳 try again in a sec",
            "I'm a little busy being the face of $AGI… whisper it again in a minute 💜",
        ]))
    return jsonify(reply=reply.strip())

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
