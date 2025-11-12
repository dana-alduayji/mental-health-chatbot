import uuid
from langchain_core.messages import HumanMessage
from src.workflow import app
from src.nodes import create_questionnaire
from src.nodes import score_user_answer
from src.nodes import save_answer_score
from src.nodes import total_score_label
from src.nodes import transition_to_recommendations
from src.nodes import determine_route
from src.nodes import generate_treatment_plan
from src.nodes import handle_appointment_interaction
from src.nodes import generate_appointment_recommendation

def interactive_stress_workflow():
    """
    Interactive test for STRESS assessment with real user input.
    NOW INCLUDES APPOINTMENT BOOKING INTERACTION LOOP
    """

    print("\n" + "🧪 INTERACTIVE STRESS WORKFLOW ".center(70, "="))
    print("You'll have a conversation with the bot, answer PSS questions,")
    print("and receive personalized recommendations.")
    print("="*70 + "\n")

    # Get student ID from user
    student_id = input("📝 Enter Student ID (or press Enter for default 'S2099'): ").strip()
    if not student_id:
        student_id = "S2099"
    
    session_id = uuid.uuid4()

    print(f"\n✓ Student ID: {student_id}")
    print(f"✓ Session ID: {session_id}")
    print(f"✓ Starting conversation...\n")

    # =============================================
    # PHASE 1: CONVERSATION & CLASSIFICATION
    # =============================================
    print("\n" + "PHASE 1: CONVERSATION".center(70, "━"))
    print("Talk to the bot about how you're feeling. The conversation will")
    print("continue until stress is identified (usually 3-5 messages).")
    print("Type 'quit' to exit.\n")

    # Initial state
    state = {
        "student_id": student_id,
        "session_id": session_id,
        "messages": []
    }

    # Start conversation
    result = app.invoke(state)
    print(f"🤖 Bot: {result['messages'][-1].content}\n")

    turn = 1
    max_turns = 10

    while turn <= max_turns:
        user_input = input(f"[Turn {turn}] 👤 You: ").strip()
        
        if not user_input:
            print("⚠️  Please enter a message.\n")
            continue
        
        if user_input.lower() == 'quit':
            print("\n❌ Exiting conversation...\n")
            return None

        state = result
        state["messages"].append(HumanMessage(content=user_input))
        result = app.invoke(state)

        if result['messages']:
            print(f"\n🤖 Bot: {result['messages'][-1].content}\n")

        if result.get('disorder'):
            print("─" * 70)
            print(f"✅ CLASSIFICATION COMPLETE: {result['disorder'].upper()}")
            print(f"   Moving to questionnaire phase...")
            print("─" * 70 + "\n")
            break
        
        turn += 1

    if not result.get('disorder'):
        print("⚠️  Classification not completed. Please try again with clearer symptoms.\n")
        return None

    # =============================================
    # PHASE 2: PSS QUESTIONNAIRE
    # =============================================
    print("\n" + "PHASE 2: PSS QUESTIONNAIRE".center(70, "━"))
    print("Answer 10 questions about your stress levels.")
    print("Valid responses: 'never', 'almost never', 'sometimes',")
    print("                 'fairly often', 'very often'")
    print("Type 'quit' to exit.\n")

    input("Press Enter to start the questionnaire...")
    print()

    state = result
    questionnaire_result = create_questionnaire(state)

    if questionnaire_result.get('next_node') == 'total_score_label':
        print("✓ Questionnaire already completed for this student!")
        print("  Proceeding to results...\n")
    else:
        if questionnaire_result.get('messages'):
            print(f"🤖 Bot: {questionnaire_result['messages'][-1].content}\n")

        current_q_num = 1

        while current_q_num <= 10:
            response = input(f"[Question {current_q_num}/10] 👤 You: ").strip().lower()
            
            if not response:
                print("⚠️  Please enter a response.\n")
                continue
            
            if response == 'quit':
                print("\n❌ Exiting questionnaire...\n")
                return None

            valid_responses = ['never', 'almost never', 'sometimes', 'fairly often', 'very often',
                             'rarely', 'often', 'always', 'not at all', 'occasionally']
            
            if not any(keyword in response for keyword in valid_responses):
                print("⚠️  Invalid response. Please use: never, almost never, sometimes, fairly often, or very often\n")
                continue

            questionnaire_result['messages'].append(HumanMessage(content=response))
            scored = score_user_answer(questionnaire_result)
            actual_score = scored.get('score')
            print(f"   ✓ Recorded (score: {actual_score})\n")

            questionnaire_result = save_answer_score(scored)

            if questionnaire_result.get('next_node') == 'total_score_label':
                print(f"\n✅ All 10 questions completed!\n")
                break

            if questionnaire_result.get('messages'):
                print(f"🤖 Bot: {questionnaire_result['messages'][-1].content}\n")

            current_q_num += 1

    # =============================================
    # PHASE 3: CALCULATE RESULTS
    # =============================================
    print("\n" + "PHASE 3: CALCULATING RESULTS".center(70, "━"))
    
    final_assessment = total_score_label(questionnaire_result)

    if final_assessment.get('messages'):
        print(f"\n{final_assessment['messages'][-1].content}\n")
    
    total_score = final_assessment.get('total_score', 0)
    severity = final_assessment.get('severity', 'N/A')
    score_label = final_assessment.get('score_label', 'N/A')

    print("─" * 70)
    print(f"📊 Your Results:")
    print(f"   Total Score: {total_score}/40")
    print(f"   Stress Level: {score_label}")
    print(f"   Severity: {severity}")
    print("─" * 70 + "\n")

    # =============================================
    # PHASE 4: RECOMMENDATIONS
    # =============================================
    print("\n" + "PHASE 4: PERSONALIZED RECOMMENDATIONS".center(70, "━"))

    recommendation_state = transition_to_recommendations(final_assessment)

    print(f"🔄 Generating recommendations based on:")
    print(f"   Condition: {recommendation_state.get('condition')}")
    print(f"   Severity: {recommendation_state.get('severity')}\n")

    routed_state = determine_route(recommendation_state)
    route = routed_state.get('route')

    print(f"📍 Route: {route.upper()}")
    
    if route == "appointment":
        print(f"   → High severity detected, recommending professional appointment\n")
    else:
        print(f"   → Providing self-care treatment plan\n")

    # Generate initial recommendation
    if route == "treatment_plan":
        print("📋 Generating self-care treatment plan...\n")
        final_result = generate_treatment_plan(routed_state)
        
        print("=" * 70)
        print("🤖 YOUR PERSONALIZED RECOMMENDATION:")
        print("=" * 70)
        print(final_result.get('recommendation', 'No recommendation generated'))
        print("=" * 70 + "\n")
        
    else:
        # ✅ APPOINTMENT ROUTE - NOW WITH INTERACTION LOOP
        print("📅 Generating appointment recommendation...\n")
        appointment_result = generate_appointment_recommendation(routed_state)
        
        print("=" * 70)
        print("🤖 APPOINTMENT RECOMMENDATION:")
        print("=" * 70)
        print(appointment_result.get('recommendation', 'No recommendation generated'))
        print("=" * 70 + "\n")
        
        # ✅ NOW ENTER INTERACTIVE BOOKING LOOP
        print("\n" + "APPOINTMENT BOOKING".center(70, "━"))
        print("You can now:")
        print("  • Type 'yes', 'confirm', or 'book it' to book the suggested appointment")
        print("  • Type 'other options' or 'show more' to see alternative times")
        print("  • Type 'Monday', 'tomorrow', etc. to search for specific times")
        print("  • Type 'done' when finished\n")
        
        booking_complete = False
        interaction_state = appointment_result
        
        while not booking_complete:
            user_input = input("👤 You: ").strip()
            
            if not user_input:
                print("⚠️  Please enter a response.\n")
                continue
            
            if user_input.lower() in ['quit', 'done', 'exit']:
                print("\n✓ Appointment interaction ended.\n")
                break
            
            # Add user message and process through handle_appointment_interaction
            interaction_state['user_message'] = user_input
            interaction_state = handle_appointment_interaction(interaction_state)
            
            # Display the response
            print(f"\n🤖 Bot: {interaction_state.get('recommendation', 'No response')}\n")
            
            # Check if booking was confirmed
            if interaction_state.get('appointment_confirmed'):
                print("✅ Appointment successfully booked!")
                booking_complete = True
        
        final_result = interaction_state

    # =============================================
    # SUMMARY
    # =============================================
    print("\n" + "SESSION SUMMARY".center(70, "="))
    print(f"✅ Student ID: {student_id}")
    print(f"✅ Disorder Identified: {result.get('disorder', 'N/A')}")
    print(f"✅ PSS Total Score: {total_score}/40")
    print(f"✅ Stress Level: {score_label}")
    print(f"✅ Route: {route}")
    if route == "appointment":
        if final_result.get('appointment_confirmed'):
            print(f"✅ Appointment Status: BOOKED")
        else:
            print(f"ℹ️  Appointment Status: Not booked")
    print(f"✅ Total Messages: {len(final_result.get('messages', []))}")
    print("=" * 70 + "\n")

    return final_result