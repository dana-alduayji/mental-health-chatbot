## Build the workflow
from langgraph.graph import StateGraph, START, END
from src.models import UnifiedState
from src.nodes import *
from src.helperfunctions import should_classify
# BUILD THE UNIFIED WORKFLOW
# ============================================
def create_unified_workflow():
    """Creates the complete 3-graph workflow with appointment interaction loop"""
    workflow = StateGraph(UnifiedState)

    # ===== GRAPH 1 NODES (Conversation & Classification) =====
    workflow.add_node("start_conversation", start_conversation)
    workflow.add_node("track", track_conversation)
    workflow.add_node("retrieve", retrieve_context)
    workflow.add_node("respond", generate_response)
    workflow.add_node("classify", classify_disorder)

    # ===== BRIDGE NODE 1: Classification → Questionnaire =====
    workflow.add_node("transition_to_questionnaire", transition_to_questionnaire)

    # ===== GRAPH 2 NODES (Questionnaire) =====
    workflow.add_node("create_questionnaire", create_questionnaire)
    workflow.add_node("ask_question", ask_question_node)
    workflow.add_node("score_answer", score_user_answer)
    workflow.add_node("save_score", save_answer_score)
    workflow.add_node("total_score_label", total_score_label)

    # ===== BRIDGE NODE 2: Questionnaire → Recommendations =====
    workflow.add_node("transition_to_recommendations", transition_to_recommendations)

    # ===== GRAPH 3 NODES (Recommendations) =====
    workflow.add_node("determine_route", determine_route)
    workflow.add_node("treatment_plan", generate_treatment_plan)
    workflow.add_node("appointment", generate_appointment_recommendation)
    workflow.add_node("handle_appointment", handle_appointment_interaction)  # ✅ NEW NODE

    # ===== GRAPH 1 EDGES (Conversation) =====
    workflow.add_edge(START, "start_conversation")
    workflow.add_edge("start_conversation", "track")

    workflow.add_conditional_edges(
        "track",
        should_classify,
        {
            "continue": "retrieve",
            "classify": "classify"
        }
    )

    workflow.add_edge("retrieve", "respond")
    workflow.add_edge("respond", END)

    # ===== BRIDGE 1: Classification → Questionnaire =====
    workflow.add_edge("classify", "transition_to_questionnaire")
    workflow.add_edge("transition_to_questionnaire", "create_questionnaire")

    # ===== GRAPH 2 EDGES (Questionnaire) =====
    workflow.add_conditional_edges(
        "create_questionnaire",
        lambda state: state.get('next_node', 'ask_question'),
        {
            'ask_question': 'ask_question',
            'total_score_label': 'total_score_label',
            'end': END
        }
    )

    workflow.add_edge("ask_question", END)
    workflow.add_edge("score_answer", "save_score")

    workflow.add_conditional_edges(
        "save_score",
        lambda state: state.get('next_node'),
        {
            'ask_question': 'ask_question',
            'total_score_label': 'total_score_label',
            'end': END
        }
    )

    # ===== BRIDGE 2: Questionnaire → Recommendations =====
    workflow.add_conditional_edges(
        "total_score_label",
        lambda state: state.get('next_node', 'end'),
        {
            'transition_to_recommendations': 'transition_to_recommendations',
            'end': END
        }
    )

    # ===== GRAPH 3 EDGES (Recommendations) =====
    workflow.add_edge("transition_to_recommendations", "determine_route")

    workflow.add_conditional_edges(
        "determine_route",
        route_by_severity,
        {
            "treatment_plan": "treatment_plan",
            "appointment": "appointment"
        }
    )

    workflow.add_edge("treatment_plan", END)
    workflow.add_edge("appointment", END)  # ✅ This now shows initial suggestion and ends
    workflow.add_edge("handle_appointment", END)  # ✅ This handles user interactions

    return workflow.compile()