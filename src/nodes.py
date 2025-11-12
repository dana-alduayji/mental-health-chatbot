from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from src.models import UnifiedState, Feedback
from src.helperfunctions import *
from src.tools import *
from typing import Literal
from datetime import datetime, timezone
from pydantic import BaseModel


def start_conversation(state: UnifiedState) -> UnifiedState:
    """Starts the conversation with a welcoming message."""
    greeting = AIMessage(content="Hello, I'm here to listen and support you. This is a safe space to share what's on your mind. How are you feeling today?")
    return {
        **state,
        "messages": [greeting],
        "iterator": 0,
        "rag_context": None,
        "workflow_stage": "conversation"
    }

def track_conversation(state: UnifiedState) -> UnifiedState:
    """Tracks the number of user inputs."""
    human_message_count = sum(1 for msg in state["messages"] if isinstance(msg, HumanMessage))
    return {**state, "iterator": human_message_count}


def retrieve_context(state: UnifiedState) -> UnifiedState:
    """Let the LLM decide whether to call RAG tool based on user message."""
    user_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
    if not user_messages:
        return {**state, "rag_context": None}

    last_user_message = user_messages[-1].content

    try:
        system_prompt = """You are a mental health therapist assistant.
Use the RAG tool to retrieve relevant mental health information from the knowledge base
when the user mentions symptoms, feelings, or concerns that could benefit from evidence-based context.

When calling the RAG tool, provide a clear search query string that describes what information to retrieve.
For example: "anxiety symptoms and coping strategies" or "depression treatment approaches"."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"User said: {last_user_message}\n\nWhat information from the knowledge base would be helpful?")
        ]

        response = llm_with_tools.invoke(messages)

        if response.tool_calls:
            tool_call = response.tool_calls[0]
            # Extract the query argument - handle both dict and string formats
            args = tool_call["args"]
            if isinstance(args, dict):
                # Use 'query' if available, otherwise use 'input', or construct from other keys
                search_query = args.get("query") or args.get("input") or last_user_message
            else:
                search_query = str(args)

            retrieved_docs = rag.invoke({"query": search_query, "k": 5})
            context = format_documents(retrieved_docs)
            print(f"✓ RAG retrieved for query: {search_query}")
            return {**state, "rag_context": context}
        else:
            return {**state, "rag_context": None}

    except Exception as e:
        print(f"⚠️  RAG retrieval error: {str(e)}")
        return {**state, "rag_context": None}

def generate_response(state: UnifiedState) -> UnifiedState:
    """Generates an empathetic, therapeutic response using RAG context."""
    system_prompt = """You are a compassionate and professional mental health therapist.
Your role is to:
- Listen actively and empathetically to the user
- Ask thoughtful, open-ended questions to understand their feelings
- Validate their emotions and experiences
- Gently explore their thoughts, feelings, and behaviors
- Use the provided knowledge base context to inform your responses with evidence-based approaches
- Do NOT diagnose or provide medical advice
- Keep responses conversational and supportive (1-3 sentences)
- Ask one follow-up question to deepen understanding

Focus on understanding if they show signs of:
- Anxiety: excessive worry, nervousness, panic, physical symptoms
- Depression: persistent sadness, loss of interest, hopelessness, fatigue
- Stress: feeling overwhelmed, tension, difficulty coping with demands
"""

    if state.get("rag_context"):
        system_prompt += f"\n\nKNOWLEDGE BASE CONTEXT:\n{state['rag_context']}\n\nUse this context to provide informed, evidence-based support while maintaining a conversational tone."

    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = llm.invoke(messages)

    return {**state, "messages": [response]}

def classify_disorder(state: UnifiedState) -> UnifiedState:
    """Analyzes conversation and classifies the disorder using RAG for ground truth."""
    conversation_summary = "\n".join([
        f"{'User' if isinstance(msg, HumanMessage) else 'Therapist'}: {msg.content}"
        for msg in state["messages"]
        if isinstance(msg, (HumanMessage, AIMessage))
    ])

    try:
        # Create a clear search query for classification
        search_query = "diagnostic criteria for anxiety depression stress mental health assessment"

        diagnostic_prompt = f"""Use the RAG tool to retrieve mental health diagnostic criteria.
Search for: {search_query}

This will help classify the following conversation:
{conversation_summary[:500]}"""

        messages = [
            SystemMessage(content="You are analyzing a mental health conversation for classification. Call the RAG tool with a search query string."),
            HumanMessage(content=diagnostic_prompt)
        ]

        response = llm_with_tools.invoke(messages)

        diagnostic_context = ""
        if response.tool_calls:
            tool_call = response.tool_calls[0]
            args = tool_call["args"]
            if isinstance(args, dict):
                query = args.get("query") or args.get("input") or search_query
            else:
                query = str(args)

            classification_docs = rag.invoke({"query": query, "k": 5})
            diagnostic_context = format_documents(classification_docs)
            print(f"✓ Classification RAG retrieved for: {query}")

    except Exception as e:
        print(f"⚠️  RAG retrieval error during classification: {str(e)}")
        diagnostic_context = ""

    classification_prompt = f"""Based on the entire conversation history, analyze the user's mental health concerns.

{diagnostic_context}

Consider the following diagnostic criteria:

ANXIETY indicators:
- Excessive worry about various things
- Restlessness or feeling on edge
- Difficulty controlling worry
- Physical symptoms (racing heart, sweating, trembling)
- Panic attacks or fear responses
- Avoidance behaviors

DEPRESSION indicators:
- Persistent sadness or low mood
- Loss of interest or pleasure in activities
- Feelings of worthlessness or guilt
- Fatigue or loss of energy
- Changes in sleep or appetite
- Hopelessness about the future
- Difficulty concentrating

STRESS indicators:
- Feeling overwhelmed by responsibilities
- Difficulty coping with demands
- Irritability or frustration
- Physical tension or headaches
- Feeling unable to manage current situation
- Recent life changes or challenges

Use the knowledge base context above as ground truth for your classification. Classify the PRIMARY concern as anxiety, depression, or stress based on the dominant pattern and evidence-based criteria."""

    messages = state["messages"] + [HumanMessage(content=classification_prompt)]
    result: Feedback = llm_structured.invoke(messages)

    type_mapping = {
    "anxiety": "GAD",
    "depression": "PHQ",
    "stress": "GAD"
    }

    try:
        supabase.table("student_questionnaire_results").insert({
            "student_id": state["student_id"],
            "type": type_mapping.get(result.condition.lower(), "GAD")  # default fallback
        }).execute()
        print("✅ Inserted classification result into Supabase.")
    except Exception as e:
        print(f"⚠️ Supabase insert failed: {e}")

    return {
        **state,
        # "disorder": result.disorder,
        "condition": result.condition,  # Map disorder to condition for graph 2
        "messages": [AIMessage(content=f"Based on our conversation and clinical evidence, I've identified your primary concern as {result.disorder}.\n\n{result.reasoning}\n\nNote: This is an AI analysis based on evidence-based mental health criteria and not a medical diagnosis. I'll now provide you with personalized recommendations and support options.")]
    }

## Transition node

def transition_to_recommendations(state: UnifiedState) -> UnifiedState:
    """
    NEW NODE: Bridges Graph 1 to Graph 2.
    Fetches actual condition/severity from database if student_id is available.
    Falls back to condition classification from conversation if DB fetch fails.
    """
    condition = state.get("condition", "stress")
    student_id = state.get("student_id")

    print(f"\n🔄 Transitioning from conversation to recommendations...")
    print(f"   condition identified from conversation: {condition}")

    # Try to fetch from database first
    if student_id:
        print(f"   Fetching assessment from database for student: {student_id}...")
        condition, severity = get_student_assessment_from_db(student_id)
        print(f"   ✓ Database assessment: {condition} ({severity})")
    else:
        # Fallback: Use conversation condition with default severity
        print(f"   No student_id provided, using conversation assessment...")
        severity_mapping = {
            "anxiety": "moderate anxiety",
            "depression": "moderate depression",
            "stress": "moderate stress"
        }
        condition = condition
        severity = severity_mapping.get(condition, "moderate stress")
        print(f"   ✓ Using mapped severity: {severity}")

    print(f"   Final assessment: {condition} at {severity} level\n")

    return {
        **state,
        "condition": condition,
        "severity": severity,
        "workflow_stage": "recommendation"
    }


def determine_route(state: UnifiedState) -> UnifiedState:
    """Determine whether student needs treatment plan or appointment."""
    severity = state["severity"].lower()
    route = SEVERITY_ROUTING.get(severity, "treatment_plan")
    print(f"✓ Route determined: {route}")
    return {**state, "route": route}

def generate_treatment_plan(state: UnifiedState) -> UnifiedState:
    """Generate self-care treatment plan for lower severity cases."""
    condition = state["condition"]
    severity = state["severity"]

    rag_context = retrieve_context_for_recommendation(condition, severity)

    system_prompt = f"""You are a compassionate mental health support assistant.

The person has been assessed with {severity} level {condition} based on our conversation.

{rag_context}

Based on the evidence-based guidelines above, provide personalized recommendations including:
- Self-care strategies specific to {condition}
- Stress management and coping techniques
- Lifestyle modifications (exercise, sleep hygiene, nutrition)
- Self-monitoring practices
- Warning signs to watch for
- When to seek additional professional support

Keep your response supportive, practical, and actionable (3 paragraphs).
Do NOT diagnose or provide medical advice."""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Provide a comprehensive self-care treatment plan for managing {condition}.")
    ]

    response = llm.invoke(messages)
    print(f"✓ Treatment plan generated")

    return {
        **state,
        "recommendation": response.content,
        "rag_context": rag_context
    }


def generate_appointment_recommendation(state: UnifiedState) -> UnifiedState:
    """Node 3b: Generate appointment recommendation for higher severity cases."""
    condition = state["condition"]
    severity = state["severity"]

    rag_context = retrieve_context_for_recommendation(condition, severity)
    nearest_slots = get_nearest_available_slot.invoke({})

    system_prompt = f"""You are a compassionate mental health support assistant with appointment booking capabilities.

The student has been assessed with {severity} level {condition}, which requires professional attention.

{rag_context}

NEAREST AVAILABLE APPOINTMENTS:
{nearest_slots}

Your response should:
1. Warmly explain why professional support is recommended at this severity level (1 sentence)
2. Present the nearest available appointment slot clearly
3. Ask the student to confirm if they'd like to book this time, or see other options
4. Provide 2-3 immediate coping strategies they can use while waiting

IMPORTANT BOOKING RULES:
- DO NOT call book_appointment tool until the student explicitly confirms
- Wait for student to say "yes", "confirm", "book it", or similar confirmation
- If they want other options, they can ask and you'll show the alternative slots listed above
- Be warm, supportive, and patient

Keep your response conversational and encouraging (2 paragraphs max)."""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Help the student understand why an appointment is needed for {condition} and guide them to book one.")
    ]

    response = llm.invoke(messages)

    print(f"✓ Appointment recommendation with nearest slot generated")

    return {
        **state,
        "recommendation": response.content,
        "rag_context": rag_context
    }

def handle_appointment_interaction(state: UnifiedState) -> UnifiedState:
    """Node 4: Interactive appointment booking/management."""
    student_id = state["student_id"]
    user_message = state.get("user_message", "")

    system_prompt = """You are an appointment booking assistant for mental health services.

Available tools:
- get_nearest_available_slot: Show nearest available appointments
- book_appointment: Book using appointment_id (ONLY after user confirms with "yes", "book it", "confirm", etc.)
- check_conflicts: Check for scheduling conflicts
- cancel_appointment: Cancel an existing appointment
- update_appointment: Reschedule an appointment (cancels old, shows new options)

CRITICAL BOOKING RULES:
1. User must explicitly confirm before booking (look for: "yes", "confirm", "book it", "okay", "sure")
2. To book, you need the appointment_id from the slot suggestion
3. If user asks for "other options" or "alternatives", show other available slots
4. If user says a specific time/date, find nearest slot to that time
5. Always be conversational and confirm what action you're taking

Example flows:
- User: "yes" → Extract appointment_id from previous suggestion → book_appointment
- User: "show me other times" → get_nearest_available_slot with more options
- User: "I prefer Monday" → get_nearest_available_slot starting from Monday"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_message)
    ]

    response = llm_with_tools_full.invoke(messages)

    tool_results = []
    booking_confirmed = False

    if response.tool_calls:
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            args = tool_call["args"]

            if tool_name in ["book_appointment", "update_appointment"]:
                args["student_id"] = student_id

            if tool_name == "get_nearest_available_slot":
                result = get_nearest_available_slot.invoke(args)
            elif tool_name == "book_appointment":
                result = book_appointment.invoke(args)
                booking_confirmed = True
            elif tool_name == "check_conflicts":
                result = check_conflicts.invoke(args)
            elif tool_name == "cancel_appointment":
                result = cancel_appointment.invoke(args)
            elif tool_name == "update_appointment":
                result = update_appointment.invoke(args)
            else:
                result = f"Unknown tool: {tool_name}"

            tool_results.append(result)

    response_text = response.content if hasattr(response, 'content') else str(response)
    full_response = response_text + "\n\n" + "\n\n".join(tool_results) if tool_results else response_text

    return {
        **state,
        "recommendation": full_response,
        "appointment_confirmed": booking_confirmed
    }

def route_by_severity(state: UnifiedState) -> str:
    """Routes to treatment_plan or appointment based on severity."""
    route = state.get("route", "treatment_plan")
    return route

## MODIFIED GRAPH 2 FUNCTIONS - Using UnifiedState with Messages

def create_questionnaire(state: UnifiedState) -> UnifiedState:
    """Initialize or resume questionnaire - returns UnifiedState with messages"""
    disorder = state.get('disorder', 'stress')
    student_id = state.get('student_id')

    print("\n" + "="*50)
    print("INITIALIZING QUESTIONNAIRE")
    print("="*50)

    questionnaire = {
        1: 'have you been upset because of something that happened unexpectedly?',
        2: 'how often have you felt that you were unable to control the important things in your life?',
        3: 'how often have you felt nervous and stressed?',
        4: 'how often have you been angered because of things that were outside of your control?',
        5: 'how often have you felt that difficulties were piling up so high that you could not overcome them?',
        6: 'how often have you found that you could not cope with all the things that you had to do?',
        7: 'how often have you felt confident about your ability to handle your personal problems?',
        8: 'how often have you felt that things were going your way?',
        9: 'how often have you been able to control irritations in your life?',
        10: 'how often have you felt that you were on top of things?'
    }

    try:
        # Reword all questions
        reword_questionnaire = {}
        print('Rewording questions for natural conversation...\n')

        for question_id, question in questionnaire.items():
            reword_question = questionnaire_reword_chain.invoke({"question": question}).content
            reword_questionnaire[f'pss{question_id}'] = reword_question

        timestamp = datetime.now(timezone.utc).isoformat()

        # Check if record exists
        exists = supabase.table('student_questionnaire_results').select('*').eq('student_id', student_id).execute()

        if not exists.data:
            # Create new record
            new_record = {
                'student_id': student_id,
                'timestamp': timestamp,
                'type': 'pss'
            }
            for i in range(1, 11):
                new_record[f'pss{i}'] = None

            supabase.table('student_questionnaire_results').insert(new_record).execute()
            print(f"✓ New record created for student {student_id}\n")

            response_text = f"Hi! I'd like to check in with you about how you've been feeling lately.\n\n{reword_questionnaire['pss1']}"
            
            return {
                **state,
                'messages': state.get('messages', []) + [AIMessage(content=response_text)],
                'current_question_id': 'pss1',
                'reword_questionnaire': reword_questionnaire,
                'next_node': 'ask_question'
            }
        else:
            # Record exists - find first unanswered
            print(f"✓ Found existing record for student {student_id}")
            record = exists.data[0]

            first_unanswered = None
            answered_count = 0

            for i in range(1, 11):
                question_key = f'pss{i}'
                if record.get(question_key) is not None:
                    answered_count += 1
                elif first_unanswered is None:
                    first_unanswered = question_key

            if first_unanswered is None:
                print("✓ All questions already answered!\n")
                response_text = "Great news! You've already completed this questionnaire. Let me calculate your results..."
                
                return {
                    **state,
                    'messages': state.get('messages', []) + [AIMessage(content=response_text)],
                    'reword_questionnaire': reword_questionnaire,
                    'next_node': 'total_score_label'
                }
            else:
                print(f"✓ Resuming from question {first_unanswered} ({answered_count}/10 completed)\n")
                response_text = f"Welcome back! Let's continue where we left off.\n\n{reword_questionnaire[first_unanswered]}"
                
                return {
                    **state,
                    'messages': state.get('messages', []) + [AIMessage(content=response_text)],
                    'current_question_id': first_unanswered,
                    'reword_questionnaire': reword_questionnaire,
                    'next_node': 'ask_question'
                }

    except Exception as e:
        error_text = f"❌ Error creating questionnaire: {e}"
        return {
            **state,
            'messages': state.get('messages', []) + [AIMessage(content=error_text)],
            'next_node': 'end'
        }


def ask_question_node(state: UnifiedState) -> UnifiedState:
    """Display question and signal that we need user input"""
    return {
        **state,
        'next_node': 'end'
    }


def score_user_answer(state: UnifiedState) -> UnifiedState:
    """
    Hybrid scoring: Try keyword matching first, fall back to LLM if needed.
    Most reliable approach.
    """
    question_id = state.get("current_question_id")
    
    # Get answer
    answer = state.get("user_answer", "")
    if not answer:
        human_messages = [msg for msg in state.get("messages", []) if isinstance(msg, HumanMessage)]
        if human_messages:
            answer = human_messages[-1].content
    
    answer = answer.lower().strip()

    print("\n" + "-"*50)
    print(f"SCORING ANSWER FOR {question_id}")
    print("-"*50)
    print(f"Answer: '{answer}'\n")

    try:
        question_num = int(question_id[3:])
        is_reverse_scoring = question_num >= 7

        # STEP 1: Try keyword matching first (fastest & most reliable)
        keyword_patterns = {
            0: ['never', 'not at all', 'no'],
            1: ['almost never', 'rarely', 'seldom', 'hardly'],
            2: ['sometimes', 'occasionally', 'once in a while'],
            3: ['fairly often', 'often', 'frequently', 'regularly'],
            4: ['very often', 'always', 'constantly', 'all the time']
        }

        matched_score = None
        for base_score, keywords in keyword_patterns.items():
            if any(keyword in answer for keyword in keywords):
                matched_score = base_score
                print(f"✓ Keyword match found: '{answer}' → base score {base_score}")
                break

        if matched_score is not None:
            # Apply reverse scoring if needed
            final_score = (4 - matched_score) if is_reverse_scoring else matched_score
            print(f"✓ Scoring type: {'REVERSE' if is_reverse_scoring else 'DIRECT'}")
            print(f"✓ Final score: {final_score}\n")
            
            return {
                **state,
                "score": final_score,
                "next_node": "save_score"
            }

        # STEP 2: No keyword match - use LLM
        print(f"⚠️ No exact keyword match, using LLM...")
        
        Score_instructions = f"""
Score this response for a stress questionnaire.

Question {question_num} uses {"REVERSE" if is_reverse_scoring else "DIRECT"} scoring.

Map to: never={"4" if is_reverse_scoring else "0"}, almost never={"3" if is_reverse_scoring else "1"}, sometimes=2, fairly often={"1" if is_reverse_scoring else "3"}, very often={"0" if is_reverse_scoring else "4"}

Response: "{answer}"

Return the numeric score (0-4) and brief reasoning.
"""

        class ScoreResponse(BaseModel):
            score: Literal[0, 1, 2, 3, 4]
            reasoning: str

        response_score = llm.with_structured_output(ScoreResponse).invoke([
            SystemMessage(content=Score_instructions),
            HumanMessage(content=answer)
        ])

        score = response_score.score if hasattr(response_score, 'score') else 2
        print(f"✓ LLM score: {score}")
        print(f"   Reasoning: {response_score.reasoning if hasattr(response_score, 'reasoning') else 'N/A'}\n")

        return {
            **state,
            "score": score,
            "next_node": "save_score"
        }

    except Exception as e:
        print(f"❌ Error scoring: {e}\n")
        return {
            **state,
            "messages": state.get('messages', []) + [AIMessage(content="Could you rephrase that?")],
            "score": 2,  # Safe default
            "next_node": "save_score"
        }


def save_answer_score(state: UnifiedState) -> UnifiedState:
    """Save score and prepare next question"""
    student_id = state.get("student_id")
    question_id = state.get("current_question_id")
    score = state.get("score")
    reword_questionnaire = state.get("reword_questionnaire", {})

    print("\n" + "-"*50)
    print(f"SAVING SCORE FOR {question_id}")
    print("-"*50)

    try:
        supabase.table("student_questionnaire_results").update(
            {question_id: score}
        ).eq("student_id", student_id).execute()

        print(f"✓ Score {score} saved successfully\n")

        exists = supabase.table("student_questionnaire_results").select("*").eq("student_id", student_id).execute()

        if not exists.data:
            error_text = "Error: Record not found"
            return {
                **state,
                "messages": state.get('messages', []) + [AIMessage(content=error_text)],
                "next_node": "end"
            }

        record = exists.data[0]
        current_num = int(question_id[3:])

        next_unanswered = None
        answered_count = 0

        for i in range(1, 11):
            key = f'pss{i}'
            if record.get(key) is not None:
                answered_count += 1
            elif next_unanswered is None and i > current_num:
                next_unanswered = key

        print(f"Progress: {answered_count}/10 questions completed")

        if next_unanswered:
            acknowledgments = [
                "Thanks for sharing that.",
                "I appreciate you telling me.",
                "Got it, thank you.",
                "Thanks for being open with me.",
                "I hear you."
            ]
            import random
            ack = random.choice(acknowledgments)

            print(f"Moving to next question: {next_unanswered}\n")

            response_text = f"{ack}\n\n{reword_questionnaire[next_unanswered]}"

            return {
                **state,
                "score": score,
                "messages": state.get('messages', []) + [AIMessage(content=response_text)],
                "current_question_id": next_unanswered,
                "reword_questionnaire": reword_questionnaire,
                "next_node": "ask_question",
            }
        else:
            print("✓ All questions completed!\n")
            response_text = "Thank you for completing all the questions. Let me calculate your results..."
            
            return {
                **state,
                "score": score,
                "messages": state.get('messages', []) + [AIMessage(content=response_text)],
                "current_question_id": None,
                "next_node": "total_score_label",
            }

    except Exception as e:
        print(f"❌ Error saving: {e}\n")
        error_text = f"Error saving your answer: {e}"
        
        return {
            **state,
            "messages": state.get('messages', []) + [AIMessage(content=error_text)],
            "next_node": "end"
        }


def total_score_label(state: UnifiedState) -> UnifiedState:
    """Calculate total score and provide assessment"""
    print("\n" + "="*50)
    print("CALCULATING FINAL RESULTS")
    print("="*50)

    student_id = state.get('student_id')
    scores = []

    try:
        exists = supabase.table('student_questionnaire_results').select('*').eq('student_id', student_id).execute()

        if exists.data and exists.data[0]['type'].lower() == "pss":
            for k, v in exists.data[0].items():
                if k.startswith('pss') and k[3:].isdigit() and v is not None:
                    scores.append(v)

        total_score = sum([x for x in scores if isinstance(x, int)])

        print(f"Individual scores: {scores}")
        print(f"Total score: {total_score}\n")

        if 0 <= total_score <= 13:
            score_label = 'Low stress'
            severity = 'low stress'
        elif 14 <= total_score <= 26:
            score_label = 'Moderate stress'
            severity = 'moderate stress'
        else:
            score_label = 'High stress'
            severity = 'high stress'

        print(f"Assessment: {score_label}\n")

        supabase.table('student_questionnaire_results').update({
            'pss_total_score': total_score,
            'pss_score_label': score_label
        }).eq('student_id', student_id).execute()

        response_text = f"""
Assessment Complete!
━━━━━━━━━━━━━━━━━━━━━
📊 Your Total Score: {total_score}/40
📈 Stress Level: {score_label}

Thank you for completing this assessment. Based on your responses, I'll now provide personalized recommendations.
"""

        return {
            **state,
            'total_score': total_score,
            'score_label': score_label,
            'severity': severity,
            'messages': state.get('messages', []) + [AIMessage(content=response_text)],
            'next_node': 'transition_to_recommendations'
        }

    except Exception as e:
        print(f"❌ Error calculating total: {e}\n")
        error_text = f'Error calculating results: {e}'
        
        return {
            **state,
            'messages': state.get('messages', []) + [AIMessage(content=error_text)],
            'next_node': 'end'
        }

def transition_to_questionnaire(state: UnifiedState) -> UnifiedState:
    """
    BRIDGE NODE 1: Connects classification to questionnaire.
    Maps disorder to questionnaire type and preserves messages.
    """
    disorder = state.get("disorder", "stress")
    student_id = state.get("student_id")
    
    print(f"\n🔄 Transitioning from classification to questionnaire...")
    print(f"   Disorder identified: {disorder}")
    print(f"   Student ID: {student_id}")
    print(f"   Messages in state: {len(state.get('messages', []))}")
    print(f"   Starting questionnaire assessment...\n")
    
    # Preserve all existing state including messages
    return {
        **state,
        "disorder": disorder,
        "workflow_stage": "questionnaire"
    }