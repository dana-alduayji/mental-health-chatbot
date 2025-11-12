import uuid
from langchain_core.messages import HumanMessage
from src.workflow import app
if __name__ == "__main__":
    print("🧠 Starting Unified Mental Health Workflow\n")

    # Initialize state
    initial_state = {
        "messages": [],
        "session_id": str(uuid.uuid4()),
        "student_id": "S2005",  # For tracking
        "disorder": None,
        "iterator": 0,
        "rag_context": None,
        "conversation_history": [],
        "workflow_stage": None
    }

    # Example: Run through conversation (Graph 1)
    print("="*70)
    print("PHASE 1: CONVERSATION & ASSESSMENT")
    print("="*70 + "\n")

    current_state = app.invoke(initial_state)
    print(f"Bot: {current_state['messages'][-1].content}\n")

    # Simulate 5 user messages
    user_inputs = [
        "I've been feeling really anxious lately",
        "Yes, I worry about everything and can't relax",
        "My heart races and I feel restless all the time",
        "It's been going on for weeks now",
        "I just can't seem to control my thoughts"
    ]

    for user_input in user_inputs:
        print(f"User: {user_input}")
        current_state["messages"].append(HumanMessage(content=user_input))
        current_state = app.invoke(current_state)
        print(f"Bot: {current_state['messages'][-1].content}\n")

    # After 5 messages, classification happens automatically
    print("\n" + "="*70)
    print("PHASE 2: RECOMMENDATIONS")
    print("="*70 + "\n")

    # The workflow automatically transitions and generates recommendations
    if current_state.get("recommendation"):
        print(f"📋 Recommendation:\n{current_state['recommendation']}\n")

    print(f"\n✅ Workflow completed!")
    print(f"   Stage: {current_state.get('workflow_stage')}")
    print(f"   Condition: {current_state.get('condition')}")
    print(f"   Route: {current_state.get('route')}")