from typing import List
from langchain_core.documents import Document
from src.tools import retrieve_treatment_info
from src.models import UnifiedState
from src.supabase import supabase
from langchain_core.prompts import ChatPromptTemplate
from src.tools import llm

SEVERITY_ROUTING = {
    "minimal depression": "treatment_plan",
    "mild depression": "treatment_plan",
    "minimal anxiety": "treatment_plan",
    "low stress": "treatment_plan",
    "moderate depression": "treatment_plan",
    "mild anxiety": "treatment_plan",
    "moderate stress": "treatment_plan",
    "moderately severe depression": "appointment",
    "moderate anxiety": "appointment",
    "high stress": "appointment",
    "severe depression": "appointment",
    "moderate to severe anxiety": "appointment"
}

def format_documents(docs: List[Document]) -> str:
    """Format retrieved documents into a readable context string."""
    if not docs:
        return "No relevant context found."
    context = "Retrieved Knowledge Base Context:\n\n"
    for i, doc in enumerate(docs, 1):
        context += f"[Document {i}]\n{doc.page_content}\n\n"
    return context

def should_classify(state: UnifiedState) -> str:
    """Determines whether to continue conversation or classify disorder."""
    iterator = state.get("iterator", 0)
    return "classify" if iterator >= 5 else "continue"






## Graph 2 Functions
questionnaire_reword_prompt = ChatPromptTemplate.from_template(
    """You are a warm, empathetic therapist assistant having a casual conversation with a student.

Your task: Transform this formal question into a natural, caring conversational statement.

CRITICAL RULES:
1. DO NOT ask a direct question - make it sound like you're checking in on them
2. DO NOT use phrases like "I'd like to ask you" or "Let me ask you"
3. DO NOT number the question or mention it's a questionnaire
4. Keep it short and conversational (1-2 sentences max)
5. Frame it around "the last month" or "recently"
6. Make it feel like a friend checking in, not a doctor diagnosing

GOOD EXAMPLES:
 BAD: "Let me ask you - in the last month, how often have you felt stressed?"
 GOOD: "Life can get pretty overwhelming sometimes. I'm curious about how things have been for you lately - have you been feeling stressed or on edge recently?"

 BAD: "I want to know how often you've felt confident."
 GOOD: "I've noticed some people have been feeling more sure of themselves lately, while others haven't. How's that been going for you over the past month?"

Now transform this question:
Question: {question}

Your conversational version (just the reworded question, nothing else):""".strip()
)

questionnaire_reword_chain = questionnaire_reword_prompt | llm


## Graph 3 Functions
def get_student_assessment_from_db(student_id: str) -> tuple[str, str]:
    """
    Get condition and severity from Supabase based on latest questionnaire.
    Returns: (condition, severity) tuple
    """
    if supabase is None:
        print("⚠️  Supabase not initialized. Using defaults.")
        return ("stress", "moderate stress")

    try:
        response = supabase.table("student_questionnaire_results") \
            .select("type, pss_score_label, phq_score_label, gad_score_label") \
            .eq("student_id", student_id) \
            .order("timestamp", desc=True) \
            .limit(1) \
            .execute()

        if not response.data:
            print(f"⚠️  No results found for student {student_id}")
            return ("stress", "moderate stress")

        result = response.data[0]
        questionnaire_type = result["type"].upper()

        type_mapping = {
            "PSS": ("stress", result.get("pss_score_label", "moderate stress")),
            "PHQ": ("depression", result.get("phq_score_label", "moderate depression")),
            "GAD": ("anxiety", result.get("gad_score_label", "mild anxiety"))
        }

        condition, severity = type_mapping.get(
            questionnaire_type,
            ("stress", "moderate stress")
        )

        return (condition, severity or "moderate stress")

    except Exception as e:
        print(f"⚠️  Error retrieving assessment: {str(e)}")
        return ("stress", "moderate stress")

def retrieve_context_for_recommendation(condition: str, severity: str) -> str:
    """
    Helper function: Retrieve RAG context for generating recommendations.
    """
    try:
        retrieved_docs = retrieve_treatment_info.invoke({
            "condition": condition,
            "severity": severity,
            "k": 5
        })

        context = format_documents(retrieved_docs)
        print(f"✓ RAG context retrieved ({len(retrieved_docs)} documents)")
        return context

    except Exception as e:
        print(f"⚠️  RAG retrieval error: {str(e)}")
        return ""
