from typing import Dict, Any

from langgraph.graph import StateGraph, START, END

from call_agent.app.graph.state import CallState
from call_agent.app.graph.context import get_recent_context

from call_agent.app.agents.memory_agent import memory_agent
from call_agent.app.agents.brain_agent import brain_agent
from call_agent.app.agents.decision_agent import decision_agent


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

    # --------------------------------------------------------
    # IMPORTANT DEPENDENCY
    #
    # Memory can run independently.
    #
    # Brain MUST finish before Decision starts.
    # --------------------------------------------------------

    graph.add_edge(
        START,
        "memory"
    )

    graph.add_edge(
        START,
        "brain"
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