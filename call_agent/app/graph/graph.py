from typing import Dict, Any

from langgraph.graph import StateGraph, START, END

from call_agent.app.graph.state import CallState
from call_agent.app.graph.context import get_recent_context

from call_agent.app.agents.memory_agent import memory_agent
from call_agent.app.agents.brain_agent import brain_agent
from call_agent.app.agents.decision_agent import decision_agent
from call_agent.app.greeting import opening_message
from call_agent.app.agents.ivr_agent import ivr_agent, ivr_respond
from call_agent.app.agents.voicemail_agent import (
    is_machine_answer,
    voicemail_agent,
)


# ============================================================
# BRAIN NODE
# ============================================================

def run_brain(
    state: CallState,
) -> Dict[str, Any]:

    conversation = state.get(
        "conversation",
        []
    )

    context = get_recent_context(
        conversation,
        limit=5,
    )

    brain_state = dict(state)

    brain_state["context"] = context

    result = brain_agent(
        brain_state
    )

    brain_output = result[
        "brain_output"
    ]

    progress = brain_output[
        "progress"
    ]

    current_stage = brain_output[
        "current_stage"
    ]

    return {
        "brain_output": brain_output,
        "progress": progress,
        "current_stage": current_stage,
        "context": context,
    }


# ============================================================
# DECISION NODE
# ============================================================

def run_decision(
    state: CallState,
) -> Dict[str, Any]:

    conversation = state.get(
        "conversation",
        []
    )

    context = get_recent_context(
        conversation,
        limit=5,
    )

    decision_state = dict(state)

    # Decision receives the UPDATED Brain state.
    decision_state["context"] = context

    result = decision_agent(
        decision_state
    )

    return {
        "decision": result[
            "decision"
        ],
        "context": context,
    }


# ============================================================
# MEMORY NODE
# ============================================================

def run_memory(
    state: CallState,
) -> Dict[str, Any]:

    result = memory_agent(
        state
    )

    return {
        "memory": result[
            "memory"
        ]
    }


# ============================================================
# CALL STATUS
# ============================================================

def finalize_call(
    state: CallState,
) -> Dict[str, Any]:

    decision = state.get(
        "decision",
        {}
    )

    present_task = decision.get(
        "present_task",
        "CONTINUE"
    )

    # IMPORTANT:
    #
    # Workflow completion NEVER ends the call.
    # Only Decision can end the call.

    call_ended = (
        present_task == "END_CALL"
    )

    return {
        "call_ended": call_ended
    }


# ============================================================
# IVR ROUTING
# ============================================================

def route_after_ivr_check(
    state: CallState,
):

    result = state.get("ivr_result", {})

    # Pressed a voicemail option and now hear the greeting: fixed line.
    if result.get("action") == "voicemail":
        return ["voicemail_agent"]

    if result.get("is_ivr") or result.get("action") == "restart":
        return ["ivr_respond"]

    return ["memory", "brain"]


# ============================================================
# GRAPH
# ============================================================

def build_graph():

    graph = StateGraph(
        CallState
    )

    graph.add_node(
        "memory",
        run_memory
    )

    graph.add_node(
        "brain",
        run_brain
    )

    graph.add_node(
        "decision",
        run_decision
    )

    graph.add_node(
        "finalize",
        finalize_call
    )

    # IVR layer: every customer turn is first checked for an automated
    # phone menu. If it is one, Anne presses keys silently (ivr_respond)
    # instead of running the normal Memory/Brain/Decision workflow.
    graph.add_node(
        "ivr_check",
        ivr_agent
    )

    graph.add_node(
        "ivr_respond",
        ivr_respond
    )

    # --------------------------------------------------------
    # IMPORTANT DEPENDENCY
    #
    # Memory can run independently.
    #
    # Brain MUST finish before Decision starts.
    # --------------------------------------------------------

    graph.add_edge(
        START,
        "ivr_check"
    )

    graph.add_conditional_edges(
        "ivr_check",
        route_after_ivr_check,
        ["memory", "brain", "ivr_respond", "voicemail_agent"],
    )

    graph.add_node(
        "voicemail_agent",
        voicemail_agent
    )

    graph.add_edge(
        "voicemail_agent",
        "finalize"
    )

    graph.add_edge(
        "ivr_respond",
        "finalize"
    )

    graph.add_edge(
        "brain",
        "decision"
    )

    # Memory does NOT block Decision.
    #
    # Therefore:
    #
    # START
    #   ├── Memory ──────────────┐
    #   │                        │
    #   └── Brain → Decision ───┤
    #
    # We only need Decision for the response path.

    graph.add_edge(
        "decision",
        "finalize"
    )

    graph.add_edge(
        "finalize",
        END
    )

    return graph.compile()


call_graph = build_graph()


# ============================================================
# START GRAPH (runs once, before the greeting)
#
# START -> voicemail_check -+-> voicemail_agent -> END   (machine)
#                           +-> greeting -------> END   (human)
# ============================================================

def voicemail_check(
    state: CallState,
) -> Dict[str, Any]:

    return {
        "is_voicemail": is_machine_answer(
            state.get("answered_by", "")
        )
    }


def route_after_voicemail_check(
    state: CallState,
) -> str:

    return "voicemail_agent" if state.get("is_voicemail") else "greeting"


def greeting(
    state: CallState,
) -> Dict[str, Any]:

    opening = opening_message(state.get("customer_name", ""))

    return {
        "decision": {
            "response": opening,
            "present_task": "CONTINUE",
        },
        "call_ended": False,
    }


def build_start_graph():

    graph = StateGraph(
        CallState
    )

    graph.add_node("voicemail_check", voicemail_check)
    graph.add_node("voicemail_agent", voicemail_agent)
    graph.add_node("greeting", greeting)

    graph.add_edge(START, "voicemail_check")

    graph.add_conditional_edges(
        "voicemail_check",
        route_after_voicemail_check,
        ["voicemail_agent", "greeting"],
    )

    graph.add_edge("voicemail_agent", END)
    graph.add_edge("greeting", END)

    return graph.compile()


start_graph = build_start_graph()
