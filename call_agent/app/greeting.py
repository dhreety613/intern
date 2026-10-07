"""Anne's scripted opening line (shared by the start graph and IVR restart)."""


def opening_message(customer_name: str) -> str:
    return (
        "Hi, this is Anne from Tensitron, calling to confirm your contact details registered with us. "
        f"Am I speaking with {customer_name}?"
    )
