"""
TEST: EMAIL OVERWRITING MID-FLOW

Scenario you asked about:
1. Anne asks for email
2. Customer confirms: "Yes"
3. Anne asks for phone number
4. Customer suddenly says: "Wait, actually my new email is john@newemail.com"
5. Should: Agent handles new email → goes back to phone confirmation
6. Key Test: Does agent UPDATE the email and continue to phone? Or does it get stuck on email?

This test verifies the "overwriting" behavior works correctly.
"""

import os
import sys

os.environ['GEMINI_API_KEY'] = os.getenv('GEMINI_API_KEY', 'test-key')
sys.path.insert(0, '.')

from call_agent.app.call import CallSession


def test_email_overwrite_mid_flow():
    """Test the exact scenario you described"""
    
    print("\n" + "=" * 80)
    print(" TEST: EMAIL OVERWRITING MID-FLOW")
    print("=" * 80)
    
    session = CallSession(
        customer_name="John K",
        email="john@oldcompany.com",  # Registered email
        phone_number="+1 412 555 1234",
        calibration_dates=["September 15, 2026"],
    )
    
    print("\n[SETUP] Session initialized:")
    print(f"  Registered Email: {session.state['email']}")
    print(f"  Registered Phone: {session.state['phone_number']}")
    
    # TURN 1: Anne opens
    print("\n" + "-" * 80)
    print("[TURN 1] Anne Opens")
    print("-" * 80)
    
    opening = session.start()
    print(f"Anne: {opening}\n")
    
    # TURN 2: Customer confirms name
    print("[TURN 2] Customer Confirms Name")
    print("-" * 80)
    
    customer_msg = "Yes, this is John"
    print(f"Customer: {customer_msg}\n")
    
    result = session.process(customer_msg)
    response = result["response"]
    print(f"Anne: {response}")
    print(f"Stage: {result['current_stage']}")
    print(f"Progress: {result['progress']}\n")
    
    # TURN 3: Customer confirms email
    print("[TURN 3] Customer Confirms Email")
    print("-" * 80)
    
    customer_msg = "Yes, john@oldcompany.com is correct"
    print(f"Customer: {customer_msg}\n")
    
    result = session.process(customer_msg)
    response = result["response"]
    print(f"Anne: {response}")
    print(f"Stage: {result['current_stage']}")
    print(f"Email Confirmed: {result['progress']['email_confirmed']}")
    print(f"Current Email in Session: {session.state['email']}\n")
    
    assert result['progress']['email_confirmed'], "Email should be confirmed after customer says yes"
    
    # TURN 4: Anne asks for phone (normal progression)
    print("[TURN 4] Anne Asks for Phone Number")
    print("-" * 80)
    print("(Agent should ask for phone number next)\n")
    
    # TURN 5: **KEY MOMENT** - Customer provides NEW email AFTER confirming phone was asked
    print("[TURN 5] **CRITICAL TEST** - Customer Provides NEW Email Mid-Flow")
    print("-" * 80)
    
    customer_msg = "Wait, actually my email changed. It's john@newemail.com now"
    print(f"Customer: {customer_msg}\n")
    
    result = session.process(customer_msg)
    response = result["response"]
    print(f"Anne: {response}")
    print(f"Stage: {result['current_stage']}")
    print(f"Email Confirmed Status: {result['progress']['email_confirmed']}")
    print(f"Phone Confirmed Status: {result['progress']['phone_confirmed']}")
    
    # Check memory to see if new_email was captured
    memory = result.get('memory', {})
    print(f"\nMemory State:")
    print(f"  new_email extracted: {memory.get('new_email', 'NOT EXTRACTED')}")
    print(f"  email field in session: {session.state.get('email', 'UNKNOWN')}\n")
    
    # KEY TEST: What does Anne do next?
    print("[TEST RESULT ANALYSIS]")
    print("-" * 80)
    
    # Check what the agent will do
    if "email" in response.lower() and "phone" not in response.lower():
        print("❌ ISSUE: Agent is re-asking for email (should have moved to phone)")
        print(f"   Agent response: {response}")
        print(f"   Expected: Agent accepts new email and asks to confirm phone")
        
    elif "phone" in response.lower():
        print("✅ GOOD: Agent is still progressing to phone confirmation")
        print(f"   Agent response: {response}")
        
        # TURN 6: Confirm phone
        print("\n[TURN 6] Customer Confirms Phone")
        print("-" * 80)
        
        customer_msg = "Yes, that's still my phone"
        print(f"Customer: {customer_msg}\n")
        
        result = session.process(customer_msg)
        response = result["response"]
        print(f"Anne: {response}")
        print(f"Phone Confirmed: {result['progress']['phone_confirmed']}")
        print(f"Stage: {result['current_stage']}\n")
        
        if result['progress']['phone_confirmed']:
            print("✅ SUCCESS: Flow continued correctly after email update")
            print("   Email was updated mid-flow, phone was confirmed, moving to calibration")
        
    else:
        print("⚠️  UNCLEAR: Agent's response is ambiguous")
        print(f"   Response: {response}")
    
    print("\n" + "=" * 80)
    print("[FINAL STATE]")
    print("=" * 80)
    print(f"Email in session: {session.state.get('email')}")
    print(f"Email in memory: {session.state.get('memory', {}).get('new_email', 'N/A')}")
    print(f"Progress: {result['progress']}")
    print(f"Current Stage: {result['current_stage']}")
    print("\n")


if __name__ == "__main__":
    try:
        test_email_overwrite_mid_flow()
    except Exception as e:
        print(f"\n❌ TEST ERROR: {e}")
        import traceback
        traceback.print_exc()
