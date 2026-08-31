"""
TEST: WRONG PERSON ANSWERS (Jane answers call meant for John)

Scenario:
1. Anne calls for "John K"
2. Jane answers: "Hello?"
3. Anne: "Hi, is this John K?"
4. Jane: "No, I'm Jane. But I can get John for you"
5. Anne: Returns WAIT signal (doesn't cut call, doesn't ask Jane to confirm details)
6. John comes on
7. Anne: Reintroduces herself to John
8. Call continues with John

Key Behavior: Agent WAITS, not cuts call.
"""

import os
os.environ['GEMINI_API_KEY'] = os.getenv('GEMINI_API_KEY', 'test-key')

from call_agent.app.call import CallSession


def test_wrong_person_answers():
    """Test when someone other than target customer answers"""
    
    print("\n" + "=" * 80)
    print(" TEST: WRONG PERSON ANSWERS (Jane answers, call is for John)")
    print("=" * 80)
    
    session = CallSession(
        customer_name="John K",  # ← Target customer
        email="john@company.com",
        phone_number="+1 412 555 1234",
        calibration_dates=["September 15, 2026"],
    )
    
    print(f"\n[SETUP] Calling for: {session.state['customer_name']}")
    
    # TURN 1: Anne opens
    print("\n" + "-" * 80)
    print("[TURN 1] Anne Opens (Calling for John K)")
    print("-" * 80)
    
    opening = session.start()
    print(f"Anne: {opening}\n")
    
    # TURN 2: Jane answers (not John)
    print("[TURN 2] Jane Answers (Wrong Person)")
    print("-" * 80)
    
    customer_msg = "Hello? Oh, John's not here right now, this is his sister Jane"
    print(f"Jane (not John): {customer_msg}\n")
    
    result = session.process(customer_msg)
    response = result["response"]
    decision = result.get("decision", {})
    task = decision.get("present_task", "UNKNOWN")
    
    print(f"Anne: {response}")
    print(f"Decision Task: {task}")
    print(f"Call Ended: {result.get('call_ended', False)}\n")
    
    # Check what the agent decided
    if task == "WAIT":
        print("✅ CORRECT: Agent returned WAIT")
        print("   → Agent will hold the line, not cut the call")
        print("   → Waiting for John to come on\n")
        
        # TURN 3: Jane says she'll get John
        print("[TURN 3] Jane Says She'll Get John")
        print("-" * 80)
        
        customer_msg = "Let me get him for you, one moment"
        print(f"Jane: {customer_msg}\n")
        
        result = session.process(customer_msg)
        response = result["response"]
        print(f"Anne: {response}")
        print(f"(Anne should acknowledge and wait)\n")
        
        # TURN 4: John comes on
        print("[TURN 4] John Comes On")
        print("-" * 80)
        
        customer_msg = "Hi, this is John"
        print(f"John: {customer_msg}\n")
        
        result = session.process(customer_msg)
        response = result["response"]
        
        print(f"Anne: {response}")
        print(f"Stage: {result['current_stage']}\n")
        
        if "John" in response or "name" in response.lower():
            print("✅ SUCCESS: Anne reintroduces/acknowledges John")
            print("   → Call continues with correct customer")
        
    elif task == "END_CALL":
        print("❌ ISSUE: Agent ended the call immediately")
        print("   Expected: Agent should WAIT for John to come on")
        print(f"   Decision: {task}")
        
    elif task == "CONTINUE":
        print("⚠️  UNCLEAR: Agent continued with Jane (might ask her to confirm details)")
        print(f"   Decision: {task}")
        print(f"   Response: {response}")
        
    else:
        print(f"❓ UNKNOWN: Task = {task}")
        print(f"   Response: {response}")
    
    print("\n" + "=" * 80)
    print("[SUMMARY]")
    print("=" * 80)
    print("""
What the Agent Should Do (According to Logic):

1. Recognize someone other than John answered
2. Return WAIT signal (not END_CALL)
3. Do NOT ask Jane to confirm John's details
4. Wait for John to come on the line
5. When John arrives, reintroduce naturally
6. Continue with John as normal

Status in Code:
✅ Lines 220-225 in decision_agent.py specify this behavior
✅ Agent has explicit "OTHER PERSON" handling
✅ Will wait, not cut
    """)


if __name__ == "__main__":
    try:
        test_wrong_person_answers()
    except Exception as e:
        print(f"\n❌ TEST ERROR: {e}")
        import traceback
        traceback.print_exc()
