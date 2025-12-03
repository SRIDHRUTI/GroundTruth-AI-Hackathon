# app.py
"""
StarbucksAI - Full application (ready-to-paste)

Key features in this file:
- Streamlit UI (chat-style) with location (browser/IP) support
- Privacy masking for PII
- Knowledge base ingestion (.txt, .json, .pdf via pdfplumber if installed)
- LangChain + Chroma RAG when OPENAI_API_KEY is present (produces structured JSON replies)
- Robust fallback local generator when LangChain/OpenAI not available
- safe_rerun() wrapper to work across Streamlit versions
- Actions (apply_coupon, get_directions, open_ticket) and interactive follow-ups
- RAGEngine.answer implemented to stitch top-k docs into context and ask ChatOpenAI for JSON-only reply
"""

import os
import re
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import streamlit as st
import streamlit.components.v1 as components
import requests
import numpy as np

# optional .env loader
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# optional PDF text extractor
try:
    import pdfplumber
except Exception:
    pdfplumber = None

# LangChain & Chroma (optional)
_have_langchain = True
try:
    from langchain.schema import Document as LC_Doc
    from langchain.embeddings import OpenAIEmbeddings
    from langchain.vectorstores import Chroma
    from langchain.chat_models import ChatOpenAI
    # Note: we won't use RetrievalQA as primary structured generator — we call ChatOpenAI directly
except Exception:
    _have_langchain = False

# -------------------------
# Configuration / Keys
# -------------------------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
WEATHER_API_KEY = os.environ.get("WEATHER_API_KEY", "")
POSITIONSTACK_API_KEY = os.environ.get("POSITIONSTACK_API_KEY", "")
PLACES_API_KEY = os.environ.get("PLACES_API_KEY", "")

KNOWLEDGE_BASE_DIR = Path("./knowledge_base")
KNOWLEDGE_BASE_DIR.mkdir(exist_ok=True)
CHROMA_PERSIST_DIR = "./chroma_db"

CONFIDENCE_HIGH = 0.75
CONFIDENCE_LOW = 0.5
RAG_TOP_K = 3

# -------------------------
# Utilities
# -------------------------
def safe_rerun():
    """Cross-version safe rerun for Streamlit."""
    try:
        # older versions
        if hasattr(st, "experimental_rerun"):
            st.experimental_rerun()
            return
    except Exception:
        pass
    try:
        # newer versions
        if hasattr(st, "rerun"):
            st.rerun()
            return
    except Exception:
        pass
    # fallback
    st.stop()

def chunk_text(text: str, chunk_size: int = 800, overlap: int = 200) -> List[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

# -------------------------
# Default knowledge docs
# -------------------------
DEFAULT_DOCS = [
    {
        "doc_id": "doc_coupons",
        "title": "Starbucks Promotions",
        "text": (
            "STAR10: Get 10% off on all hot beverages for Rewards members. "
            "WARM20: Get 20% off on hot drinks when temperature is below 20°C. "
            "COOL15: Get 15% off on cold drinks when temperature is above 25°C. "
            "NEWMEMBER: 20% off your first order when you join Starbucks Rewards."
        ),
        "source": "promotions.txt",
        "tags": ["coupon", "promotions"]
    }
]

# -------------------------
# Privacy masking
# -------------------------
class PrivacyMasker:
    PATTERNS = [
        ('CARD', r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b'),
        ('EMAIL', r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
        ('ORDER_ID', r'\b(?:ORD|ORDER|SB)[-_]?\d{4,10}\b'),
        ('PHONE', r'\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}\b'),
    ]

    def __init__(self):
        self.mask_map = {}
        self.counter = {}

    def mask_text(self, text: str) -> Tuple[str, Dict]:
        self.mask_map = {}
        self.counter = {name: 0 for name, _ in self.PATTERNS}
        masked = text
        for pii_type, pattern in self.PATTERNS:
            matches = re.findall(pattern, masked, re.IGNORECASE)
            for match in matches:
                if match not in self.mask_map.values():
                    self.counter[pii_type] += 1
                    token = f"<MASKED_{pii_type}_{self.counter[pii_type]}>"
                    self.mask_map[token] = match
                    masked = masked.replace(match, token)
        return masked, self.mask_map

    def get_masked_count(self) -> Dict[str, int]:
        return {k: v for k, v in self.counter.items() if v > 0}

# -------------------------
# Weather API (WeatherAPI.com)
# -------------------------
def get_weather(lat: float, lon: float) -> Optional[Dict]:
    """Fetch current weather (WeatherAPI.com)."""
    if not WEATHER_API_KEY:
        return None
    try:
        url = "http://api.weatherapi.com/v1/current.json"
        params = {"key": WEATHER_API_KEY, "q": f"{lat},{lon}", "aqi": "no"}
        r = requests.get(url, params=params, timeout=5)
        if r.status_code == 200:
            j = r.json()
            return {
                "temp_c": j["current"]["temp_c"],
                "feels_like": j["current"]["feelslike_c"],
                "condition": j["current"]["condition"]["text"],
                "humidity": j["current"]["humidity"],
                "city": j["location"]["name"],
                "region": j["location"]["region"],
                "country": j["location"]["country"]
            }
    except Exception as e:
        print("Weather API error:", e)
    return None

# -------------------------
# Geolocation helper (browser + IP fallback)
# -------------------------
def get_browser_location(key="browser_location", use_ip_fallback=True):
    """
    Try browser geolocation (JS). If unavailable, optionally use IP-based geolocation.
    Stores result in st.session_state[key].
    """
    if key in st.session_state and st.session_state.get(key) is not None:
        return st.session_state.get(key)

    js = """
    <script>
    (function() {
      function send(msg) {
        const el = document.getElementById("streamlit-geoloc-output");
        if (el) { el.innerText = JSON.stringify(msg); }
      }
      function success(pos) { send({lat: pos.coords.latitude, lon: pos.coords.longitude}); }
      function error(err) { send({error: err ? err.message : "unknown"}); }
      if (navigator && navigator.geolocation) {
        navigator.geolocation.getCurrentPosition(success, error, {timeout:10000});
      } else {
        send({error: "geolocation_not_supported"});
      }
    })();
    </script>
    <div id="streamlit-geoloc-output" style="display:none"></div>
    """
    try:
        result = components.html(js, height=0)
        if result:
            try:
                payload = json.loads(result)
                if "lat" in payload and "lon" in payload:
                    st.session_state[key] = {"lat": float(payload["lat"]), "lon": float(payload["lon"])}
                    return st.session_state[key]
            except Exception:
                pass
        # try to read the hidden div
        read_js = """
        <script>
          try {
            const el = document.getElementById("streamlit-geoloc-output");
            if (el && el.innerText) document.write(el.innerText);
            else document.write("");
          } catch (e) { document.write(""); }
        </script>
        """
        read_result = components.html(read_js, height=0)
        if read_result:
            try:
                payload = json.loads(read_result)
                if "lat" in payload and "lon" in payload:
                    st.session_state[key] = {"lat": float(payload["lat"]), "lon": float(payload["lon"])}
                    return st.session_state[key]
            except Exception:
                pass
    except Exception:
        pass

    # IP fallback
    if use_ip_fallback:
        try:
            r = requests.get("https://ipapi.co/json/", timeout=3)
            if r.status_code == 200:
                j = r.json()
                lat = j.get("latitude") or j.get("lat")
                lon = j.get("longitude") or j.get("lon")
                if lat and lon:
                    st.session_state[key] = {"lat": float(lat), "lon": float(lon), "city": j.get("city")}
                    return st.session_state[key]
        except Exception as e:
            print("IP geolocation fallback failed:", e)

    return st.session_state.get(key, None)

# -------------------------
# Simulated nearby stores (demo)
# -------------------------
def get_nearby_starbucks(lat: float, lon: float, city: str = "") -> List[Dict]:
    city_stores = {
        "hyderabad": [
            {"name": "Starbucks Jubilee Hills", "distance_m": 180, "address": "Road No. 36, Jubilee Hills, Hyderabad", "rating": 4.4},
            {"name": "Starbucks Banjara Hills", "distance_m": 350, "address": "Road No. 12, Banjara Hills, Hyderabad", "rating": 4.5},
            {"name": "Starbucks Inorbit Mall", "distance_m": 520, "address": "Inorbit Mall, Madhapur, Hyderabad", "rating": 4.3}
        ],
        "new york": [
            {"name": "Starbucks Times Square", "distance_m": 120, "address": "1585 Broadway, New York, NY", "rating": 4.1},
            {"name": "Starbucks Reserve Roastery", "distance_m": 280, "address": "61 9th Ave, New York, NY", "rating": 4.7}
        ]
    }
    city_lower = (city or "").lower()
    stores = []
    for k, lst in city_stores.items():
        if k in city_lower or city_lower in k:
            stores = lst.copy()
            break
    if not stores:
        stores = [{"name": f"Starbucks {city or 'Downtown'}", "distance_m": 200, "address": f"Main Street, {city or 'Your City'}", "rating": 4.3}]
    for s in stores:
        s["open"] = True
    return sorted(stores, key=lambda x: x["distance_m"])

# -------------------------
# LangChain + Chroma helpers
# -------------------------
def build_chroma_from_docs(docs: List[Dict], persist_dir: str = CHROMA_PERSIST_DIR):
    """Build Chroma vectorstore from docs using OpenAIEmbeddings."""
    if not _have_langchain:
        print("LangChain/Chroma not available.")
        return None, None
    if not OPENAI_API_KEY:
        print("OPENAI_API_KEY not set - cannot build Chroma embeddings.")
        return None, None
    try:
        lc_docs = []
        for d in docs:
            meta = {"doc_id": d.get("doc_id"), "source": d.get("source"), "title": d.get("title")}
            lc_docs.append(LC_Doc(page_content=d.get("text", ""), metadata=meta))
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        vect = Chroma.from_documents(documents=lc_docs, embedding=embeddings, persist_directory=persist_dir)
        vect.persist()
        retriever = vect.as_retriever(search_kwargs={"k": RAG_TOP_K})
        return vect, retriever
    except Exception as e:
        print("Chroma build failed:", e)
        return None, None

# -------------------------
# Simple VectorStore fallback (TF-ish)
# -------------------------
class SimpleVectorStore:
    def __init__(self):
        self.documents = []
        self.embeddings = []

    def _embed(self, text: str) -> np.ndarray:
        words = re.findall(r'\w+', text.lower())
        vec = np.zeros(512, dtype=float)
        for w in words:
            vec[hash(w) % 512] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def add_documents(self, docs: List[Dict]):
        for doc in docs:
            text = f"{doc.get('title','')} {doc.get('text','')} {' '.join(doc.get('tags',[]))}"
            self.documents.append(doc)
            self.embeddings.append(self._embed(text))

    def query(self, query_text: str, top_k: int = 3) -> List[Dict]:
        if not self.documents:
            return []
        qv = self._embed(query_text)
        scores = [(i, float(np.dot(qv, emb))) for i, emb in enumerate(self.embeddings)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return [{**self.documents[i], "score": s} for i, s in scores[:top_k]]

# -------------------------
# Knowledge base loader (.txt, .json, .pdf)
# -------------------------
def load_knowledge_base() -> List[Dict]:
    docs = DEFAULT_DOCS.copy()
    # load txt
    for p in sorted(KNOWLEDGE_BASE_DIR.glob("*.txt")):
        try:
            txt = p.read_text(encoding="utf-8")
            docs.append({"doc_id": f"doc_{p.stem}", "title": p.stem.replace("_", " ").title(), "text": txt, "source": p.name, "tags": [p.stem]})
        except Exception as e:
            print("Error loading txt:", p, e)
    # load json
    for p in sorted(KNOWLEDGE_BASE_DIR.glob("*.json")):
        try:
            j = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(j, list):
                docs.extend(j)
            elif isinstance(j, dict):
                docs.append(j)
        except Exception as e:
            print("Error loading json:", p, e)
    # load pdfs (if pdfplumber installed)
    if pdfplumber:
        for p in sorted(KNOWLEDGE_BASE_DIR.glob("*.pdf")):
            try:
                with pdfplumber.open(p) as pdf:
                    all_text = "\n".join([page.extract_text() or "" for page in pdf.pages])
                chunks = chunk_text(all_text, chunk_size=400, overlap=80)
                for idx, ch in enumerate(chunks):
                    docs.append({"doc_id": f"doc_{p.stem}_part{idx}", "title": p.stem.replace("_", " ").title(), "text": ch, "source": p.name, "tags": [p.stem]})
            except Exception as e:
                print("Error loading pdf:", p, e)
    else:
        for p in sorted(KNOWLEDGE_BASE_DIR.glob("*.pdf")):
            print(f"PDF found ({p.name}) but pdfplumber not installed. Install pdfplumber to extract text.")
    return docs

# -------------------------
# Intent extraction (simple rules with OpenAI optional later)
# -------------------------
def extract_intent(message: str) -> Dict:
    msg = message.lower()
    if any(w in msg for w in ["cold", "freezing", "chilly", "winter"]):
        return {"intent": "cold_weather", "slots": {}}
    if any(w in msg for w in ["hot", "warm", "heat", "summer"]):
        return {"intent": "hot_weather", "slots": {}}
    if any(w in msg for w in ["where", "nearest", "nearby", "location", "store", "find"]):
        return {"intent": "find_store", "slots": {}}
    if any(w in msg for w in ["menu", "drink", "food", "ice cream", "i wanna eat", "i want to eat"]):
        return {"intent": "menu_inquiry", "slots": {}}
    if any(w in msg for w in ["coupon", "promo", "offer", "code", "discount"]):
        return {"intent": "coupon_inquiry", "slots": {}}
    if any(w in msg for w in ["refund", "money back"]):
        return {"intent": "refund_request", "slots": {}}
    if any(w in msg for w in ["order", "missing", "late", "wrong"]):
        return {"intent": "order_issue", "slots": {}}
    if any(w in msg for w in ["card", "charge", "payment", "billing"]):
        return {"intent": "billing_issue", "slots": {}}
    if any(w in msg for w in ["reward", "star", "points", "loyalty"]):
        return {"intent": "rewards_inquiry", "slots": {}}
    if any(w in msg for w in ["hello", "hi", "hey"]):
        return {"intent": "greeting", "slots": {}}
    if any(w in msg for w in ["thanks", "thank you"]):
        return {"intent": "thanks", "slots": {}}
    return {"intent": "general", "slots": {}}

# -------------------------
# Local fallback generator (friendly)
# -------------------------
def generate_response_local_improved(intent: Dict, weather: Optional[Dict], stores: List[Dict],
                                     docs: List[Dict], user_profile: Dict, message: str) -> Dict:
    """
    Structured fallback response used when RAG/LLM is unavailable.
    Returns keys: reply, actions, citations, confidence, follow_ups, cards
    """
    intent_type = intent.get("intent", "general")
    nearest = stores[0] if stores else None
    temp = weather.get("temp_c") if weather else None
    city = weather.get("city", "") if weather else ""
    response = {"reply": "", "actions": [], "citations": [], "confidence": 0.85, "follow_ups": [], "cards": []}

    def coupon_for_temp(t):
        if t is None:
            return None
        if t < 20:
            return "WARM20"
        if t > 25:
            return "COOL15"
        return None

    # Greeting
    if intent_type == "greeting":
        response["reply"] = f"👋 Hello! I can help with ☕ suggestions, 📍 finding stores, or 🎟️ promos. What would you like?"
        response["follow_ups"] = ["Find nearest store", "Show hot drinks", "Show promotions"]
        response["confidence"] = 0.95
        return response

    # Menu
    if intent_type in ("menu_inquiry", "menu_hot", "menu_cold"):
        # try extract price-bearing lines from docs
        items = []
        for d in docs:
            for ln in re.split(r'[\r\n]+|\.\s+', d.get("text", "")):
                ln = ln.strip()
                if ln and re.search(r'₹|\$|\b\d{2,4}\b', ln):
                    items.append(ln)
        if not items:
            for d in docs[:3]:
                items.extend([l.strip() for l in re.split(r'[\r\n]+|\.\s+', d.get("text", ""))][:6])
        if items:
            md_lines = ["🧾 **Menu (sample):**"]
            for it in items[:10]:
                md_lines.append(f"- {it}")
            coupon = coupon_for_temp(temp)
            if coupon:
                md_lines.append(f"\n🎟️ It's {temp:.1f}°C — consider **{coupon}** for a discount.")
                response["actions"].append({"type": "apply_coupon", "params": {"coupon": coupon}, "label": f"✅ Apply {coupon}"})
            response["reply"] = "\n".join(md_lines)
            response["citations"] = [{"doc_id": d.get("doc_id"), "source": d.get("source")} for d in docs[:2]]
            response["follow_ups"] = ["Recommend a hot drink", "Recommend a cold drink", "Nearest store"]
            return response
        response["reply"] = "We have hot drinks, cold drinks and food. What would you like — hot, cold, or food?"
        response["follow_ups"] = ["Hot drinks", "Cold drinks", "Food options"]
        return response

    # Cold feeling
    if intent_type == "cold_weather":
        # If local temp is warm, ask clarification
        if temp is not None and temp >= 22.0:
            response["reply"] = (f"❄️ You're feeling cold but it's {temp:.1f}°C in {city}. "
                                 "Do you want a hot drink recommendation to warm up, or are you asking about cold-weather promos?")
            response["follow_ups"] = ["Recommend a hot drink", "Show cold-weather promos", "Nearest store"]
            response["confidence"] = 0.85
            return response
        # otherwise recommend hot drinks
        suggestion = "Hot Chocolate ☕️"
        for d in docs:
            if "hot" in d.get("title", "").lower() or "hot" in " ".join(d.get("tags", [])).lower():
                match = re.search(r'([A-Za-z &]+)\s*[-–]\s*(₹?\$?\d+)', d.get("text", ""))
                if match:
                    suggestion = f"{match.group(1).strip()} ({match.group(2)}) ☕️"
                    break
        parts = []
        if temp is not None:
            parts.append(f"🥶 It's {temp:.1f}°C in {city}.")
        if nearest:
            parts.append(f"📍 Nearby: **{nearest['name']}** ({nearest['distance_m']} m).")
        parts.append(f"I recommend **{suggestion}** to warm up!")
        coupon = coupon_for_temp(temp)
        if coupon == "WARM20":
            parts.append(f"Use **{coupon}** for 20% off hot drinks.")
            response["actions"].append({"type": "apply_coupon", "params": {"coupon": coupon}, "label": f"✅ Apply {coupon}"})
        response["reply"] = " ".join(parts)
        response["follow_ups"] = ["Get directions", "Show more hot drinks", "Apply coupon"]
        if nearest:
            response["cards"].append({
                "type": "store",
                "name": nearest["name"],
                "distance_m": nearest["distance_m"],
                "address": nearest.get("address", ""),
                "rating": nearest.get("rating", None),
                "open": nearest.get("open", True)
            })
        return response

    # Hot weather
    if intent_type == "hot_weather":
        suggestion = "Cold Brew 🧊"
        for d in docs:
            if "cold" in d.get("title", "").lower() or "cold" in " ".join(d.get("tags", [])).lower():
                match = re.search(r'([A-Za-z &]+)\s*[-–]\s*(₹?\$?\d+)', d.get("text", ""))
                if match:
                    suggestion = f"{match.group(1).strip()} ({match.group(2)}) 🧊"
                    break
        parts = []
        if temp is not None:
            parts.append(f"🔥 It's {temp:.1f}°C in {city}.")
        if nearest:
            parts.append(f"📍 Nearest: **{nearest['name']}** ({nearest['distance_m']} m).")
        parts.append(f"Try **{suggestion}** to cool down.")
        coupon = coupon_for_temp(temp)
        if coupon == "COOL15":
            parts.append(f"Use **{coupon}** for 15% off cold drinks.")
            response["actions"].append({"type": "apply_coupon", "params": {"coupon": coupon}, "label": f"✅ Apply {coupon}"})
        response["reply"] = " ".join(parts)
        response["follow_ups"] = ["Show cold drinks", "Apply coupon", "Nearest store hours"]
        return response

    # Find store
    if intent_type == "find_store":
        if nearest:
            md = f"📍 **Nearest Starbucks:** {nearest['name']}\n\n• Distance: {nearest['distance_m']} m\n• Address: {nearest['address']}\n• Rating: {nearest.get('rating','N/A')}/5\n• Status: {'Open ✅' if nearest.get('open') else 'Closed ⛔'}"
            response["reply"] = md
            response["actions"].append({"type": "get_directions", "params": {"store": nearest['name']}, "label": "🗺️ Get directions"})
            response["follow_ups"] = ["Show other nearby stores", "Open store hours"]
            response["cards"].append({
                "type": "store",
                "name": nearest["name"],
                "distance_m": nearest["distance_m"],
                "address": nearest.get("address", ""),
                "rating": nearest.get("rating", None),
                "open": nearest.get("open", True)
            })
            return response
        else:
            response["reply"] = "I don't have your location yet. Please click **Use my current location** or enter coordinates on the left. 📍"
            response["follow_ups"] = ["Use my current location", "Set Hyderabad location"]
            return response

    # Coupon inquiry
    if intent_type == "coupon_inquiry":
        lines = [
            "🎟️ **Current promotions:**",
            "- STAR10 — 10% off for Rewards members",
            "- WARM20 — 20% off hot drinks (below 20°C)",
            "- COOL15 — 15% off cold drinks (above 25°C)"
        ]
        response["reply"] = "\n".join(lines)
        response["actions"].append({"type": "apply_coupon", "params": {"coupon": "STAR10"}, "label": "✅ Apply STAR10"})
        response["follow_ups"] = ["How do I join Rewards?", "Show eligible items"]
        return response

    # Refund / billing
    if intent_type in ("refund_request", "billing_issue"):
        response["reply"] = "Billing and refunds are sensitive. I can open a support ticket and mask any PII. Would you like me to create one? 🎫"
        response["actions"].append({"type": "open_ticket", "params": {"issue": intent_type}, "label": "📝 Open support ticket"})
        response["confidence"] = 0.4
        response["follow_ups"] = ["Yes, open ticket", "No, cancel"]
        return response

    # Default fallback
    response["reply"] = "I can help with menu recommendations, finding nearby stores, coupons, and refunds. What would you like to do? ☕️"
    response["follow_ups"] = ["Show menu", "Find nearest store", "Apply coupon"]
    return response

# -------------------------
# LLM structured reply helper (for RAGEngine)
# -------------------------
def call_llm_for_structured_reply(llm, user_query: str, context_docs: List, weather: Optional[Dict], nearest_store: Optional[Dict], max_tokens: int = 400) -> Optional[Dict]:
    """
    Call ChatOpenAI to produce a JSON-only reply based on top-k retrieved docs and context.
    Expects llm to be a LangChain ChatOpenAI instance with a callable interface.
    """
    def make_doc_snippet(d):
        # d may be a langchain Document with page_content & metadata, or a dict
        text = ""
        meta = {}
        if hasattr(d, "page_content"):
            text = d.page_content or ""
            meta = getattr(d, "metadata", {}) or {}
        elif isinstance(d, dict):
            text = d.get("text", "") or d.get("page_content", "")
            meta = {"doc_id": d.get("doc_id"), "source": d.get("source")}
        snippet = text.strip().replace("\n", " ")
        snippet = snippet[:400]
        header = f"[{meta.get('doc_id') or meta.get('source') or 'doc'}]"
        return f"{header} {snippet}"

    snippets = "\n\n".join([make_doc_snippet(d) for d in context_docs])

    ctx_lines = []
    if weather:
        ctx_lines.append(f"Weather: {weather.get('temp_c')}°C, {weather.get('condition','')} in {weather.get('city','')}.")
    if nearest_store:
        ctx_lines.append(f"Nearest store: {nearest_store.get('name')} ({nearest_store.get('distance_m')}m), address: {nearest_store.get('address','')}.")
    context_block = "\n".join(ctx_lines)

    system_prompt = f"""
You are a helpful Starbucks customer support assistant. Use the provided documents and context to answer the user's question.
Return ONLY valid JSON (no explanation) that follows this exact schema:

{{
  "reply": "<string - user-facing markdown-friendly reply, max 500 chars>",
  "actions": [ {{ "type":"apply_coupon|get_directions|open_ticket|call_support", "label":"<button label>", "params":{{}} }} ],
  "citations": [ "<doc_id or source id>" ],
  "follow_ups": [ "<short suggestion text>" ],
  "confidence": <float between 0.0 and 1.0>
}}

Rules:
- Use the docs for factual claims and include their doc_id values in "citations" when you reference them.
- Use weather & location to recommend hot drinks when cold, cold drinks when hot.
- If the docs do not contain the answer, say so briefly in the "reply" field.
- Keep tone friendly and include 1-2 emojis when appropriate.
"""

    user_prompt = f"""
User question: {user_query}

Documents (top-k):
{snippets if snippets else 'None'}

Context:
{context_block if context_block else 'None'}

Instructions: Produce the JSON as described in the system prompt above.
"""

    try:
        # Compose messages using LangChain message objects if available
        from langchain.schema import SystemMessage, HumanMessage
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        resp = llm(messages=messages, max_tokens=max_tokens)
        # Extract text content robustly
        text = ""
        try:
            # Many langchain versions return AIMessage-like object with content attribute
            if hasattr(resp, "content"):
                text = resp.content
            elif isinstance(resp, list) and len(resp) > 0 and hasattr(resp[0], "message"):
                text = resp[0].message.content
            elif isinstance(resp, str):
                text = resp
            else:
                text = str(resp)
        except Exception:
            text = str(resp)

        # Extract JSON between first '{' and last '}'
        if "{" in text:
            json_str = text[text.find("{"): text.rfind("}")+1]
        else:
            json_str = text

        parsed = json.loads(json_str)
        # Basic validation and normalization
        if isinstance(parsed, dict) and "reply" in parsed:
            parsed["confidence"] = float(parsed.get("confidence", 0.85))
            if not isinstance(parsed.get("actions", []), list):
                parsed["actions"] = []
            if not isinstance(parsed.get("citations", []), list):
                parsed["citations"] = []
            if not isinstance(parsed.get("follow_ups", []), list):
                parsed["follow_ups"] = []
            # clamp confidence
            parsed["confidence"] = max(0.0, min(1.0, parsed["confidence"]))
            return parsed
    except Exception as e:
        print("LLM structured reply error:", e)
    return None

# -------------------------
# RAGEngine with structured LLM JSON
# -------------------------
class RAGEngine:
    def __init__(self, docs: List[Dict]):
        self.docs = docs
        self.chroma = None
        self.retriever = None
        self.llm = None
        if _have_langchain and OPENAI_API_KEY:
            try:
                self.chroma, self.retriever = build_chroma_from_docs(self.docs, persist_dir=CHROMA_PERSIST_DIR)
                if self.retriever:
                    # Use ChatOpenAI wrapper
                    self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.0, openai_api_key=OPENAI_API_KEY)
                    print("✓ LangChain + ChatOpenAI ready")
            except Exception as e:
                print("LangChain init error:", e)
        else:
            if not _have_langchain:
                print("LangChain not installed; falling back to local generator.")
            if not OPENAI_API_KEY:
                print("OPENAI_API_KEY missing; falling back to local generator.")

    def answer(self, query: str, weather: Optional[Dict], nearest_store: Optional[Dict]) -> Optional[Dict]:
        """
        Retrieve top-k docs and ask LLM to synthesize a structured JSON reply.
        Returns parsed dict or None.
        """
        if not self.retriever or not self.llm:
            return None

        try:
            docs = self.retriever.get_relevant_documents(query)
        except Exception:
            docs = []

        # Send top-K docs to LLM for structured JSON response
        structured = call_llm_for_structured_reply(llm=self.llm, user_query=query, context_docs=docs[:RAG_TOP_K], weather=weather, nearest_store=nearest_store)
        if structured:
            # fill citations if empty
            if not structured.get("citations"):
                doc_ids = []
                for d in docs[:RAG_TOP_K]:
                    md = getattr(d, "metadata", {}) or (d if isinstance(d, dict) else {})
                    doc_ids.append(md.get("doc_id") or md.get("source") or "unknown")
                structured["citations"] = doc_ids
            return structured

        # fallback: return None so caller can use local generator
        return None

# -------------------------
# Metrics
# -------------------------
class Metrics:
    def __init__(self):
        self.data = {"queries": 0, "intents": {}, "actions": 0, "pii_masked": 0}
    def record(self, intent: str, pii: int, actions: int):
        self.data["queries"] += 1
        self.data["intents"][intent] = self.data["intents"].get(intent, 0) + 1
        self.data["pii_masked"] += pii
        self.data["actions"] += actions
    def action_executed(self):
        self.data["actions"] += 1

# -------------------------
# Main orchestrator
# -------------------------
class StarbucksAI:
    def __init__(self):
        self.masker = PrivacyMasker()
        self.metrics = Metrics()
        self.docs = load_knowledge_base()
        self.rag = RAGEngine(self.docs)
        self.simple_vs = SimpleVectorStore()
        self.simple_vs.add_documents(self.docs)
        print(f"✓ Loaded {len(self.docs)} documents into knowledge base")

    def process(self, message: str, user_profile: Dict = None, lat: float = None, lon: float = None) -> Dict:
        # 1. Mask PII
        masked_msg, mask_map = self.masker.mask_text(message)
        pii_count = sum(self.masker.get_masked_count().values())

        # 2. Intent
        intent = extract_intent(masked_msg)

        # 3. Weather
        weather = get_weather(lat, lon) if lat and lon else None

        # 4. Nearby stores
        city = weather.get("city", "") if weather else ""
        stores = get_nearby_starbucks(lat or 0, lon or 0, city) if lat and lon else []

        # 5. RAG query
        query = f"{intent.get('intent', '')} {masked_msg}"
        if weather:
            query += f" {weather.get('temp_c', '')} degrees"

        # 6. Prefer structured LLM + retriever if available
        if self.rag and self.rag.retriever and self.rag.llm:
            rag_result = self.rag.answer(query, weather, stores[0] if stores else None)
            if rag_result:
                self.metrics.record(intent=intent.get("intent", "unknown"), pii=pii_count, actions=len(rag_result.get("actions", [])))
                return {"response": rag_result, "intent": intent, "weather": weather, "stores": stores, "docs": [], "pii_masked": pii_count}

        # 7. Fallback to simple vector retrieval and local improved generator
        docs = self.simple_vs.query(query, top_k=RAG_TOP_K)
        baseline = generate_response_local_improved(intent=intent, weather=weather, stores=stores, docs=docs, user_profile=user_profile or {}, message=masked_msg)
        self.metrics.record(intent=intent.get("intent", "unknown"), pii=pii_count, actions=len(baseline.get("actions", [])))
        return {"response": baseline, "intent": intent, "weather": weather, "stores": stores, "docs": docs, "pii_masked": pii_count}

# -------------------------
# Action executor
# -------------------------
def execute_action(action: Dict) -> str:
    action_type = action.get("type", "")
    params = action.get("params", {})
    if action_type == "apply_coupon":
        return f"✅ Coupon **{params.get('coupon')}** applied! (demo)"
    if action_type == "get_directions":
        return f"📍 Opening directions to {params.get('store','Starbucks')}... (demo)"
    if action_type == "open_ticket":
        ticket_id = f"SB-{int(time.time()) % 100000}"
        return f"🎫 Support ticket **{ticket_id}** created! Our team will contact you within 24 hours."
    if action_type == "call_support":
        return "📞 Connecting you to Starbucks support... (demo)"
    return "Action completed."

# -------------------------
# Quick reply handler
# -------------------------
def handle_quick_reply(ai_instance: StarbucksAI, quick_text: str):
    st.session_state.messages.append({"role": "user", "content": quick_text})
    with st.chat_message("user", avatar="👤"):
        st.markdown(quick_text)

    with st.spinner("Thinking..."):
        loc = st.session_state.location
        result = ai_instance.process(message=quick_text, user_profile=st.session_state.profile, lat=loc.get("lat"), lon=loc.get("lon"))

    response = result["response"]
    reply = response.get("reply", "Sorry, I couldn't help with that.")
    actions = response.get("actions", [])
    with st.chat_message("assistant", avatar="☕"):
        if response.get("confidence", 1) < CONFIDENCE_LOW:
            st.warning("⚠️ I may be uncertain about this. Consider human review.")
        st.markdown(reply)
        if response.get("citations"):
            if isinstance(response["citations"][0], dict):
                sources = ", ".join([c.get("doc_id", c.get("source", str(c))) for c in response["citations"]])
            else:
                sources = ", ".join(response["citations"])
            st.caption(f"📚 Sources: {sources}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": reply,
        "actions": actions,
        "cards": response.get("cards", []),
        "follow_ups": response.get("follow_ups", [])
    })

# -------------------------
# Streamlit UI
# -------------------------
def main():
    st.set_page_config(page_title="StarbucksAI Support", page_icon="☕", layout="wide")

    # initialize session state
    if "ai" not in st.session_state:
        st.session_state.ai = StarbucksAI()
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "memory" not in st.session_state:
        st.session_state.memory = []
    if "location" not in st.session_state:
        st.session_state.location = {"lat": None, "lon": None}
    if "profile" not in st.session_state:
        st.session_state.profile = {}
    if "pending_action" not in st.session_state:
        st.session_state.pending_action = None

    ai = st.session_state.ai

    # Sidebar (no quick starters)
    with st.sidebar:
        st.image("https://upload.wikimedia.org/wikipedia/en/thumb/d/d3/Starbucks_Corporation_Logo_2011.svg/1200px-Starbucks_Corporation_Logo_2011.svg.png", width=100)
        st.title("StarbucksAI")
        st.divider()
        st.subheader("📍 Your Location")

        browser_loc = get_browser_location()
        if browser_loc:
            try:
                st.success(f"Detected: {browser_loc.get('lat'):.5f}, {browser_loc.get('lon'):.5f}")
            except Exception:
                st.success("Detected location")
            if st.button("Use detected location", use_container_width=True):
                st.session_state.location = {"lat": float(browser_loc["lat"]), "lon": float(browser_loc["lon"])}
                safe_rerun()
        else:
            if st.button("Use my current location", use_container_width=True):
                _ = get_browser_location(use_ip_fallback=True)
                safe_rerun()

        col1, col2 = st.columns(2)
        with col1:
            lat = st.number_input("Latitude", value=st.session_state.location.get("lat") or 0.0, format="%.5f", label_visibility="collapsed")
        with col2:
            lon = st.number_input("Longitude", value=st.session_state.location.get("lon") or 0.0, format="%.5f", label_visibility="collapsed")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🇮🇳 Hyderabad", use_container_width=True):
                st.session_state.location = {"lat": 17.385, "lon": 78.4867}
                safe_rerun()
        with col2:
            if st.button("🇺🇸 New York", use_container_width=True):
                st.session_state.location = {"lat": 40.7128, "lon": -74.006}
                safe_rerun()

        try:
            if lat != 0.0 or lon != 0.0:
                st.session_state.location = {"lat": float(lat), "lon": float(lon)}
        except Exception:
            pass

        if st.session_state.location["lat"]:
            st.success(f"📍 {st.session_state.location['lat']:.5f}, {st.session_state.location['lon']:.5f}")

        st.divider()
        st.subheader("👤 Your Profile")
        name = st.text_input("Name (Optional)", placeholder="Your name")
        tier = st.selectbox("Rewards Status", ["Not a member", "Green", "Gold"])
        profile = {}
        if name:
            profile["name"] = name
        if tier != "Not a member":
            profile["loyalty_tier"] = tier.lower()
        st.session_state.profile = profile

        st.divider()
        st.subheader("📊 Session Stats")
        st.caption(f"Messages: {len(st.session_state.messages)}")
        st.caption(f"Intents: {len(ai.metrics.data['intents'])}")
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.memory = []
            safe_rerun()

    # Main chat area
    st.title("☕ Starbucks Customer Support")
    st.caption("Powered by AI — personalized to your location & preferences")

    # Render messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="☕" if msg["role"] == "assistant" else "👤"):
            st.markdown(msg["content"])
            if msg["role"] == "assistant":
                # actions
                if "actions" in msg and msg["actions"]:
                    cols = st.columns(len(msg["actions"]))
                    for i, action in enumerate(msg["actions"]):
                        act_label = action.get("label", "Run")
                        if cols[i].button(act_label, key=f"act_{hash(str(action)+msg['content'][:20])}"):
                            st.session_state.pending_action = action
                # cards
                if "cards" in msg and msg["cards"]:
                    for card in msg["cards"]:
                        if card.get("type") == "store":
                            st.markdown("---")
                            c1, c2 = st.columns([3, 1])
                            with c1:
                                st.markdown(f"**📍 {card.get('name')}**  ")
                                st.markdown(f"{card.get('address')}  ")
                                st.caption(f"Distance: {card.get('distance_m')} m • Rating: {card.get('rating','N/A')}")
                            with c2:
                                if card.get("open"):
                                    st.success("Open ✅")
                                else:
                                    st.error("Closed ⛔")
                            st.markdown("---")
                # follow-ups (quick reply chips)
                if "follow_ups" in msg and msg["follow_ups"]:
                    st.write("")
                    fr_cols = st.columns(min(4, len(msg["follow_ups"])))
                    for i, fq in enumerate(msg["follow_ups"]):
                        if fr_cols[i].button(fq, key=f"fq_{hash(fq)+i}"):
                            handle_quick_reply(ai, fq)

    # Pending action confirmation
    if st.session_state.pending_action:
        action = st.session_state.pending_action
        st.info(f"**Confirm:** {action.get('label')}")
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button("✅ Yes, proceed", use_container_width=True):
                result = execute_action(action)
                ai.metrics.action_executed()
                st.session_state.messages.append({"role": "assistant", "content": result})
                st.session_state.pending_action = None
                safe_rerun()
        with col2:
            if st.button("❌ Cancel", use_container_width=True):
                st.session_state.pending_action = None
                safe_rerun()

    # Chat input
    if prompt := st.chat_input("Ask me anything about Starbucks..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        # Process the message
        with st.spinner("Thinking..."):
            loc = st.session_state.location
            result = ai.process(message=prompt, user_profile=st.session_state.profile, lat=loc.get("lat"), lon=loc.get("lon"))

        response = result["response"]
        reply = response.get("reply", "I'm not sure how to help with that.")
        actions = response.get("actions", [])
        cards = response.get("cards", [])
        follow_ups = response.get("follow_ups", [])

        # Update memory
        st.session_state.memory.append({"role": "user", "text": prompt})
        st.session_state.memory.append({"role": "assistant", "text": reply})
        st.session_state.memory = st.session_state.memory[-6:]

        # Display assistant reply
        with st.chat_message("assistant", avatar="☕"):
            if response.get("confidence", 1) < CONFIDENCE_LOW:
                st.warning("⚠️ I'm not fully confident about this. Consider speaking to a human.")
            st.markdown(reply)
            if response.get("citations"):
                if isinstance(response["citations"][0], dict):
                    sources = ", ".join([c.get("doc_id", c.get("source", str(c))) for c in response["citations"]])
                else:
                    sources = ", ".join(response["citations"])
                st.caption(f"📚 Sources: {sources}")
            if response.get("follow_up"):
                st.info(f"💡 {response['follow_up']}")

        # Append assistant message with metadata for UI (actions/cards/follow_ups)
        st.session_state.messages.append({
            "role": "assistant",
            "content": reply,
            "actions": actions,
            "cards": cards,
            "follow_ups": follow_ups
        })

        # Rerun to show interactive buttons
        safe_rerun()

# -------------------------
# Entrypoint
# -------------------------
if __name__ == "__main__":
    print("\n" + "="*60)
    print("☕ StarbucksAI - RAG (structured LLM replies) + friendly fallback")
    print("="*60)
    print(f"LangChain+Chroma available: {'✓' if _have_langchain else '✗ (install langchain & chromadb)'}")
    print(f"OpenAI API: {'✓' if OPENAI_API_KEY else '✗ Set OPENAI_API_KEY'}")
    print(f"Weather API: {'✓' if WEATHER_API_KEY else '✗ Set WEATHER_API_KEY'}")
    print(f"Knowledge Base: {KNOWLEDGE_BASE_DIR}")
    print("="*60 + "\n")
    main()
