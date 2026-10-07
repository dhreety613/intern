"""
Test script to verify Anne's full conversation flow locally.

This simulates the voice agent behavior WITHOUT actual audio/TTS/STT,
just using the CallSession directly to test the conversation logic.

Expected flow:
1. Anne: "Hi, this is Anne from Tensitron..."
2. Customer confirms/denies name
3. If yes → confirm name → email → phone → calibration date reminder
4. Wait for "nothing more" or similar
5. Anne: "Thank you..." → end call
"""

from call_agent.app.call import CallSession


def test_full_flow():
    """Test the complete conversation flow."""

    print("\n" + "=" * 70)
    print("ANNE VOICE AGENT - LOCAL FLOW TEST")
    print("=" * 70 + "\n")

    # Create session with test customer data
    session = CallSession(
        customer_name="Jane K",
        email="john@company.com",
        phone_number="+1 412 555 1234",
        calibration_dates=["September 15, 2026"],
    )

    # 1. Anne's opening
    print("\n--- TURN 1: Anne Opens ---")
    opening = session.start()
    print(f"Anne: {opening}\n")

    # 2. Customer confirms name (YES)
    print("--- TURN 2: Customer Says Yes ---")
    customer_input = "Yes, this is Jane"
    print(f"Customer: {customer_input}\n")

    result = session.process(customer_input)
    response = result.get("response", "")
    print(f"Anne: {response}")
    print(f"[Task: {result.get('decision', {}).get('present_task', 'UNKNOWN')}]")
    print(f"[Stage: {result.get('current_stage', 'UNKNOWN')}]\n")

    # 3. Confirm email
    print("--- TURN 3: Confirm Email ---")
    customer_input = "Yes that's correct"
    print(f"Customer: {customer_input}\n")

    result = session.process(customer_input)
    response = result.get("response", "")
    print(f"Anne: {response}")
    print(f"[Task: {result.get('decision', {}).get('present_task', 'UNKNOWN')}]")
    print(f"[Stage: {result.get('current_stage', 'UNKNOWN')}]\n")

    # 4. Confirm phone
    print("--- TURN 4: Confirm Phone ---")
    customer_input = "Yes that's my number"
    print(f"Customer: {customer_input}\n")

    result = session.process(customer_input)
    response = result.get("response", "")
    print(f"Anne: {response}")
    print(f"[Task: {result.get('decision', {}).get('present_task', 'UNKNOWN')}]")
    print(f"[Stage: {result.get('current_stage', 'UNKNOWN')}]\n")

    # 5. Acknowledge calibration reminder
    print("--- TURN 5: Acknowledge Calibration ---")
    customer_input = "Understood, I have that noted"
    print(f"Customer: {customer_input}\n")

    result = session.process(customer_input)
    response = result.get("response", "")
    print(f"Anne: {response}")
    print(f"[Task: {result.get('decision', {}).get('present_task', 'UNKNOWN')}]")
    print(f"[Stage: {result.get('current_stage', 'UNKNOWN')}]\n")

    # 6. Customer says nothing else needed
    print("--- TURN 6: Customer Indicates Done ---")
    customer_input = "That's all I need, thank you"
    print(f"Customer: {customer_input}\n")

    result = session.process(customer_input)
    response = result.get("response", "")
    call_ended = result.get("call_ended", False)
    print(f"Anne: {response}")
    print(f"[Task: {result.get('decision', {}).get('present_task', 'UNKNOWN')}]")
    print(f"[Call Ended: {call_ended}]\n")

    print("=" * 70)
    if call_ended:
        print("✓ FLOW COMPLETE - Call ended as expected")
    else:
        print("✗ ISSUE - Call did not end properly")
    print("=" * 70 + "\n")

    # Print final state
    print("\n--- FINAL CALL STATE ---")
    print(f"Conversation turns: {len(session.state.get('conversation', []))}")
    print(f"Progress: {session.state.get('progress', {})}")
    print(f"Call Ended: {session.state.get('call_ended', False)}\n")


if __name__ == "__main__":
    test_full_flow()
