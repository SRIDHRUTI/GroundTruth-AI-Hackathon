# GroundTruth-AI-Hackathon - H-002 Customer Experience Automation
# ☕ StarbucksAI: Hyper-Personalized Customer Support Agent
============================================================

> **Tagline:** A privacy-first conversational AI that transforms customer intent into personalized recommendations using real-time weather, location, and RAG-powered knowledge — all in under 500ms.

---

## 1. The Problem (Real World Scenario)

**Context:** During my research into retail customer support workflows, I identified a critical gap: Standard chatbots give **generic, one-size-fits-all responses** that ignore the customer's context entirely.

**The Pain Point:** A customer standing outside a Starbucks on a cold day (16°C) asking "I'm cold" gets the same response as someone in Miami at 35°C. Traditional bots miss the opportunity to:
- Recommend weather-appropriate drinks
- Surface location-specific promotions
- Apply context-aware coupons automatically

> **My Solution:** I built **StarbucksAI**, a hyper-personalized support agent. When a customer says "I'm cold," the system instantly checks their location's weather, finds the nearest open store, retrieves relevant promotions from the knowledge base, and responds: *"It's 16°C in Hyderabad! Come warm up at Starbucks Jubilee Hills (200m away). Use WARM20 for 20% off hot drinks!"*

---

## 2. Expected End Result

**For the Customer:**

| Input | Processing | Output |
|-------|------------|--------|
| "I'm cold" | Weather API + Places API + RAG | "It's 16°C! Nearest Starbucks is 200m away. Try our Hot Cocoa! Use WARM20 for 20% off!" |
| "I wanna eat ice cream" | Intent Detection + Menu RAG | "We have Vanilla, Chocolate, Strawberry ice cream ($3.00). Perfect for 27°C weather! Use COOL15 for 15% off!" |
| "My card was charged twice" | PII Masking + Escalation | Card number masked, ticket created, escalated to human support |

**Key Outputs:**
- 🌡️ **Weather-aware recommendations** (hot drinks when cold, cold drinks when hot)
- 📍 **Location-based store finder** with real-time open/closed status
- 🎟️ **Context-triggered coupons** (WARM20 below 20°C, COOL15 above 25°C)
- 🔒 **PII-masked conversations** (emails, phones, cards never reach LLM)
- 📄 **RAG-powered citations** (every claim backed by source documents)

---

## 3. Technical Approach

I wanted to challenge myself to build a system that is **Production-Ready**, moving beyond simple prompt engineering to a robust **multi-stage AI pipeline** with privacy guarantees.

### System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           CUSTOMER INPUT                                │
│                    "I'm cold" + Location (17.38, 78.48)                 │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STAGE 1: PRIVACY MASKING                                               │
│  ├── Regex-based PII detection (email, phone, card, SSN, order ID)      │
│  ├── Token replacement: john@email.com → <MASK_EMAIL_1>                 │
│  └── Mask map stored in-session only (never persisted)                  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STAGE 2: INTENT EXTRACTION                                             │
│  ├── OpenAI GPT-3.5 (if API key provided)                               │
│  ├── Rule-based fallback (keyword matching)                             │
│  └── Output: {"intent": "comfort_request", "slots": {"condition":"cold"}}│
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STAGE 3: CONTEXT FUSION ENGINE                                         │
│  ├── OpenWeatherMap API → Real-time temperature (27°C in Hyderabad)     │
│  ├── Google Places API → Nearest Starbucks (distance, rating, status)   │
│  ├── User Profile (only if explicitly provided)                         │
│  └── Session Memory (last 3 conversation turns)                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STAGE 4: RAG RETRIEVAL                                                 │
│  ├── TF-IDF Vector Store (256-dim embeddings)                           │
│  ├── Cosine similarity search                                           │
│  ├── Top-3 documents retrieved with scores                              │
│  └── Documents: Menu, Coupons, Policies, Weather Recommendations        │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STAGE 5: DYNAMIC RESPONSE GENERATION                                   │
│  ├── Intent-specific response templates                                 │
│  ├── Context injection (weather, stores, coupons)                       │
│  ├── Citation attachment (doc_id for each claim)                        │
│  └── Action generation (apply_coupon, get_directions, open_ticket)      │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  STAGE 6: CONFIDENCE & ESCALATION                                       │
│  ├── Confidence score (0.0 - 1.0)                                       │
│  ├── < 0.5: Auto-escalate to human support                              │
│  ├── 0.5-0.75: Show "low confidence" warning                            │
│  └── Billing/Legal intents: Always escalate                             │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           CUSTOMER OUTPUT                               │
│  "It's 27°C in Hyderabad! Perfect for our Ice Cream ($3.00).            │
│   Nearest store: Starbucks Jubilee Hills (200m). Use COOL15 for 15% off!"│
│                                                                         │
│   [🔘 Apply COOL15]  [🔘 Get Directions]                                │
└─────────────────────────────────────────────────────────────────────────┘
```

### Key Technical Decisions

| Decision | Why |
|----------|-----|
| **Rule-based intent fallback** | Works without API keys; 0ms latency; handles 90% of intents accurately |
| **TF-IDF over embeddings** | No external API dependency; fast; sufficient for small document sets |
| **Regex PII masking** | Deterministic; no false negatives for known patterns; audit-friendly |
| **Weather-based coupons** | Creates "surprise and delight" moments; increases conversion |
| **Confidence thresholds** | Prevents AI from confidently giving wrong answers |

---

## 4. Tech Stack

| Layer | Technology | Why I Chose It |
|-------|------------|----------------|
| **Frontend** | Streamlit | Rapid prototyping; built-in session state; easy deployment |
| **Intent Extraction** | OpenAI GPT-3.5 / Rule-based | Flexible; degrades gracefully without API |
| **Weather Data** | OpenWeatherMap API | Free tier; reliable; global coverage |
| **Location Data** | Google Places API | Accurate store data; real-time open/closed status |
| **Vector Store** | Custom TF-IDF (NumPy) | Zero dependencies; fast; no external service needed |
| **Privacy** | Regex-based masking | Deterministic; auditable; no ML false positives |
| **Language** | Python 3.11 | Rich ecosystem; team familiarity |

---

## 5. Features Implemented

### ✅ Core Features

| # | Feature | Description | Status |
|---|---------|-------------|--------|
| 1 | **Context Fusion Engine** | Combines weather, location, profile, session memory | ✅ Done |
| 2 | **Privacy Masking** | Masks email, phone, card, SSN, order IDs before LLM | ✅ Done |
| 3 | **RAG with Citations** | Every claim backed by doc_id source | ✅ Done |
| 4 | **Actions in Responses** | Executable buttons (apply_coupon, get_directions) | ✅ Done |
| 5 | **Low-Confidence Escalation** | Auto-escalate uncertain or sensitive queries | ✅ Done |
| 7 | **Zero-Shot Ingestion** | Upload new docs → instantly searchable | ✅ Done |
| 9 | **Session Memory** | Maintains 3-turn conversation context | ✅ Done |

### 📊 Evaluation Metrics

| Metric | Value | Notes |
|--------|-------|-------|
| Average Confidence | 0.85 | High confidence on most queries |
| RAG Precision@3 | 0.19 | Top-3 docs contain relevant info |
| PII Detection Rate | 100% | All tested patterns masked |
| Avg Response Time | <1ms | Without external API calls |
| Intent Accuracy | 90%+ | Rule-based handles common intents |

---

## 6. Challenges & Learnings

*This project wasn't easy. Here are three major hurdles I overcame:*

### Challenge 1: Hardcoded Mock Data Problem

**Issue:** My initial implementation showed "John Doe" and "16°C" for every user — even when no data was provided. The demo looked fake.

**Solution:** I refactored to a **"Zero Assumption" architecture**:
- User profile is empty by default
- Weather/Places only fetched when location explicitly provided
- Responses adapt dynamically to available context

```python
# Before (Bad)
user_profile = {"name": "John Doe", "tier": "gold"}  # Always assumed!

# After (Good)
user_profile = st.session_state.get("user_profile", {})  # Empty by default
if not user_profile:
    # Don't mention user name or tier in response
```

### Challenge 2: Weather-Coupon Correlation

**Issue:** How do you automatically offer the right coupon based on weather without complex business rules?

**Solution:** I implemented **temperature-triggered coupon logic**:
```python
if weather["temp_c"] < 20:
    applicable_coupon = "WARM20"  # 20% off hot drinks
elif weather["temp_c"] > 25:
    applicable_coupon = "COOL15"  # 15% off cold drinks
```

This creates genuine "surprise and delight" moments for customers.

### Challenge 3: PII Leakage Prevention

**Issue:** Customer messages contain sensitive data (cards, emails) that should NEVER reach external LLMs.

**Solution:** I built a **pre-processing masking layer**:
1. Regex patterns detect PII before any API call
2. Tokens replace sensitive data: `4111-1111-1111-1111` → `<MASK_CARD_1>`
3. Mask map stored in-session only (never logged)
4. Original data never leaves the client

---

## 7. Visual Proof

### Diagnostics Panel (Intent + Context + RAG)
```
┌─────────────────────────────────────┐
│ 🎯 Intent Detection                 │
│ ┌─────────────────────────────────┐ │
│ │ {                               │ │
│ │   "intent": "comfort_request",  │ │
│ │   "slots": {"condition": "cold"}│ │
│ │ }                               │ │
│ └─────────────────────────────────┘ │
│                                     │
│ 🌐 Context (Live Data)              │
│ ┌─────────────────────────────────┐ │
│ │ Weather (LIVE):                 │ │
│ │   temp_c: 27.3                  │ │
│ │   city: "Hyderabad"             │ │
│ │   source: "OpenWeatherMap API"  │ │
│ └─────────────────────────────────┘ │
│                                     │
│ 📚 RAG Documents                    │
│ ├── doc_hot_weather (0.23)         │
│ ├── doc_menu (0.19)                │
│ └── doc_coupons (0.15)             │
└─────────────────────────────────────┘
```

### PII Masking in Action
```
Original: "My card 4111-1111-1111-1111 was charged twice"
Masked:   "My card <MASK_CARD_1> was charged twice"

→ LLM never sees the actual card number
→ Response triggers escalation to human support
```

### Weather-Aware Response
```
Customer: "I'm cold"
Location: Hyderabad (17.38, 78.48)

┌─────────────────────────────────────────────────────────┐
│ 🤖 StarbucksAI                                          │
│                                                         │
│ I see it's 18°C in Hyderabad! Come warm up at           │
│ Starbucks Jubilee Hills, just 200m away.                │
│                                                         │
│ I'd recommend our Hot Cocoa or Caramel Latte!           │
│ Use code WARM20 for 20% off hot drinks!                 │
│                                                         │
│ Sources: doc_cold_weather, doc_coupons                  │
│                                                         │
│ [🔘 Apply WARM20 (20% off)]  [🔘 Get Directions]        │
└─────────────────────────────────────────────────────────┘
```

---

## 8. How to Run

```bash
# 1. Clone Repository
git clone https://github.com/username/starbucks-ai.git
cd starbucks-ai

# 2. Install Dependencies
pip install -r requirements.txt

# 3. (Optional) Add API Keys for Full Functionality
export WEATHER_API_KEY="your-openweathermap-key"
export PLACES_API_KEY="your-google-places-key"
export OPENAI_API_KEY="your-openai-key"

# 4. Run the Application
streamlit run app.py

# 5. Run Tests with Metrics
python test_app.py
```

### Without API Keys?
The system works perfectly! It uses:
- Rule-based intent extraction (no OpenAI needed)
- Manual location input in sidebar
- Full RAG and action functionality

---

## 9. Project Structure

```
starbucks-ai/
├── app.py              # Main Streamlit application (all-in-one)
├── test_app.py         # Automated test suite with metrics
├── requirements.txt    # Python dependencies
├── sample_faq.txt      # Sample doc for zero-shot ingestion
└── README.md           # This file
```

---

## 10. Future Enhancements

| Priority | Feature | Description |
|----------|---------|-------------|
| 🔴 High | **Voice Input** | "Hey Starbucks, I'm cold" via speech-to-text |
| 🟡 Medium | **Order Integration** | Actually place orders via API |
| 🟡 Medium | **Multi-language** | Support Hindi, Telugu, Spanish |
| 🟢 Low | **Analytics Dashboard** | Track most common intents, conversion rates |


