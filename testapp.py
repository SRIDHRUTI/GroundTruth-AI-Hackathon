#!/usr/bin/env python3
"""
Test Script for StarbucksAI
===========================
"""

import sys
sys.path.insert(0, '.')

from app import (
    PrivacyMasker,
    VectorStore,
    StarbucksAI,
    extract_intent,
    get_nearby_starbucks,
    DEFAULT_DOCS
)

def print_header(title):
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def print_result(name, passed, details=""):
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"{status} | {name}")
    if details:
        print(f"       └─ {details}")

def test_privacy_masking():
    print_header("TEST: Privacy Masking")
    
    masker = PrivacyMasker()
    
    # Test email
    text1 = "My email is john@starbucks.com"
    masked1, _ = masker.mask_text(text1)
    email_ok = "john@starbucks.com" not in masked1 and "<MASKED_EMAIL>" in masked1
    print_result("Email masking", email_ok, masked1)
    
    # Test phone
    masker2 = PrivacyMasker()
    text2 = "Call me at 555-123-4567"
    masked2, _ = masker2.mask_text(text2)
    phone_ok = "555-123-4567" not in masked2
    print_result("Phone masking", phone_ok, masked2)
    
    # Test card
    masker3 = PrivacyMasker()
    text3 = "My card 4111-1111-1111-1111 was charged"
    masked3, _ = masker3.mask_text(text3)
    card_ok = "4111" not in masked3
    print_result("Card masking", card_ok, masked3)
    
    return all([email_ok, phone_ok, card_ok])

def test_intent_extraction():
    print_header("TEST: Intent Extraction")
    
    tests = [
        ("I'm cold", ["cold_weather"]),
        ("It's so hot today", ["hot_weather"]),
        ("I want a frappuccino", ["menu_cold"]),
        ("Where is the nearest store?", ["find_store"]),
        ("I want a refund", ["refund_request"]),
        ("What coupons are available?", ["coupon_inquiry"]),
        ("Hello!", ["greeting"]),
        ("Tell me about rewards", ["rewards_inquiry"]),
    ]
    
    all_passed = True
    for msg, expected_intents in tests:
        result = extract_intent(msg)
        intent = result.get("intent", "")
        passed = intent in expected_intents
        print_result(f"'{msg}' -> {expected_intents[0]}", passed, f"Got: {intent}")
        all_passed = all_passed and passed
    
    return all_passed

def test_rag_retrieval():
    print_header("TEST: RAG Retrieval")
    
    store = VectorStore()
    store.add_documents(DEFAULT_DOCS)
    
    # Test hot drinks query
    results1 = store.query("hot coffee latte", top_k=3)
    hot_found = any("hot" in d.get("doc_id", "").lower() or "menu" in d.get("doc_id", "").lower() for d in results1)
    print_result("Hot drinks query", hot_found, f"Top: {results1[0]['doc_id']}")
    
    # Test cold drinks query
    results2 = store.query("cold frappuccino iced", top_k=3)
    cold_found = any("cold" in d.get("doc_id", "").lower() or "menu" in d.get("doc_id", "").lower() for d in results2)
    print_result("Cold drinks query", cold_found, f"Top: {results2[0]['doc_id']}")
    
    # Test coupons query
    results3 = store.query("discount coupon promo", top_k=3)
    coupon_found = any("coupon" in d.get("doc_id", "").lower() for d in results3)
    print_result("Coupon query", coupon_found, f"Top: {results3[0]['doc_id']}")
    
    return all([hot_found, cold_found, coupon_found])

def test_nearby_stores():
    print_header("TEST: Nearby Stores")
    
    # Test Hyderabad
    stores1 = get_nearby_starbucks(17.385, 78.4867, "Hyderabad")
    hyd_ok = len(stores1) > 0 and ("Hyderabad" in stores1[0].get("address", "") or "Jubilee" in stores1[0].get("name", ""))
    print_result("Hyderabad stores", hyd_ok, f"Found: {stores1[0]['name']}")
    
    # Test New York
    stores2 = get_nearby_starbucks(40.7128, -74.006, "New York")
    ny_ok = len(stores2) > 0
    print_result("New York stores", ny_ok, f"Found: {stores2[0]['name']}")
    
    # Test sorting by distance
    sorted_ok = all(stores1[i]["distance_m"] <= stores1[i+1]["distance_m"] for i in range(len(stores1)-1))
    print_result("Stores sorted by distance", sorted_ok)
    
    return all([hyd_ok, ny_ok, sorted_ok])

def test_full_pipeline():
    print_header("TEST: Full Pipeline")
    
    ai = StarbucksAI()
    
    # Test without location
    result1 = ai.process("Where is the nearest Starbucks?")
    no_loc_ok = "location" in result1["response"]["reply"].lower() or "share" in result1["response"]["reply"].lower()
    print_result("Store query without location asks for location", no_loc_ok)
    
    # Test with location
    result2 = ai.process("I'm cold", lat=17.385, lon=78.4867)
    cold_ok = "warm" in result2["response"]["reply"].lower() or "hot" in result2["response"]["reply"].lower()
    has_store = "Starbucks" in result2["response"]["reply"]
    print_result("Cold weather response recommends warm drinks", cold_ok)
    print_result("Response includes store info", has_store)
    
    # Test coupon suggestion
    has_coupon = any(a.get("type") == "apply_coupon" for a in result2["response"].get("actions", []))
    print_result("Coupon action suggested", has_coupon)
    
    # Test PII masking
    result3 = ai.process("My email is test@test.com")
    pii_ok = result3["pii_masked"] > 0
    print_result("PII detected and masked", pii_ok)
    
    return all([no_loc_ok, cold_ok, has_store])

def test_no_hardcoded_data():
    print_header("TEST: No Hardcoded User Data")
    
    ai = StarbucksAI()
    
    # Test that responses don't mention fake user names
    result = ai.process("Hello!")
    no_john = "john" not in result["response"]["reply"].lower()
    no_fake_email = "@email.com" not in result["response"]["reply"]
    
    print_result("No 'John' in response", no_john)
    print_result("No fake email in response", no_fake_email)
    
    return all([no_john, no_fake_email])

def run_all_tests():
    print("\n" + "☕"*30)
    print("  STARBUCKS AI - TEST SUITE")
    print("☕"*30)
    
    results = []
    results.append(("Privacy Masking", test_privacy_masking()))
    results.append(("Intent Extraction", test_intent_extraction()))
    results.append(("RAG Retrieval", test_rag_retrieval()))
    results.append(("Nearby Stores", test_nearby_stores()))
    results.append(("Full Pipeline", test_full_pipeline()))
    results.append(("No Hardcoded Data", test_no_hardcoded_data()))
    
    print_header("TEST SUMMARY")
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        print(f"  {'✅' if result else '❌'} {name}")
    
    print(f"\n  🏆 {passed}/{total} tests passed!")
    
    print("\n" + "="*60)
    print("  To run the app:")
    print("  1. Set WEATHER_API_KEY (from weatherapi.com)")
    print("  2. Run: streamlit run app.py")
    print("="*60 + "\n")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)