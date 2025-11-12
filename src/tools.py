from typing import List
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.tools import tool
from langchain_community.vectorstores import FAISS
from datetime import datetime
from src.models import Feedback
from src.supabase import supabase

llm = ChatOpenAI(temperature=0.5)

@tool
def rag(query: str, k: int = 5) -> List[Document]:
    """
    Load the FAISS index and retrieve the top related documents.

    Args:
        query: The search query string to find relevant documents
        k: Number of documents to retrieve (default: 5)
    """
    folder_path = "/content/faiss_index"
    db = FAISS.load_local(
        folder_path,
        embeddings=OpenAIEmbeddings(),
        allow_dangerous_deserialization=True
    )
    return db.similarity_search(query, k)

llm_with_tools = llm.bind_tools([rag])
llm_structured = llm.with_structured_output(Feedback)

## Graph 2 Tools

@tool
def get_nearest_available_slot(datetime_str: str = None, num_suggestions: int = 3) -> str:
    """Get the nearest available appointment slots starting from requested time or now."""
    try:
        from dateutil import parser
        from datetime import timedelta

        if datetime_str:
            dt = parser.parse(datetime_str, fuzzy=True, default=datetime.now())
        else:
            dt = datetime.now()

        target_date = dt.date().isoformat()
        target_time = dt.time().strftime("%H:%M:%S")

        result = (
            supabase.table('appointments')
            .select("appointment_id, appointment_time, appointment_date")
            .eq("status", "Available")
            .gte("appointment_date", target_date)
            .order("appointment_date")
            .order("appointment_time")
            .limit(10)
            .execute()
        )

        if not result.data:
            return "No available slots found."

        available_slots = []
        for record in result.data:
            slot_date = record['appointment_date']
            slot_time = record['appointment_time']

            if slot_date == target_date and slot_time < target_time:
                continue

            available_slots.append({
                'id': record['appointment_id'],
                'datetime': f"{slot_date} at {slot_time}",
                'date': slot_date,
                'time': slot_time
            })

            if len(available_slots) >= num_suggestions:
                break

        if not available_slots:
            return "No available slots found after the requested time."

        nearest = available_slots[0]
        response = f"📅 Nearest available slot: {nearest['datetime']} (ID: {nearest['id']})"

        if len(available_slots) > 1:
            response += "\n\nOther available options:"
            for slot in available_slots[1:]:
                response += f"\n  • {slot['datetime']} (ID: {slot['id']})"

        return response

    except Exception as e:
        return f"Error getting nearest slots: {e}"

@tool
def book_appointment(appointment_id: str, student_id: str) -> str:
    """Book an appointment using the appointment ID after user confirmation."""
    try:
        result = (
            supabase.table('appointments')
            .update({
                "status": "Booked",
                "student_id": student_id
            })
            .eq("appointment_id", appointment_id)
            .eq("status", "Available")
            .execute()
        )

        if result.data:
            appointment = result.data[0]
            return f"✓ Appointment confirmed! Booked for {appointment.get('appointment_date')} at {appointment.get('appointment_time')}. Appointment ID: {appointment_id}"
        else:
            return f"✗ Unable to book. This slot may no longer be available."

    except Exception as e:
        return f"Error booking appointment: {e}"

@tool
def check_conflicts(datetime_str: str) -> str:
    """Check for conflicts within 1 hour of the specified time."""
    try:
        from dateutil import parser
        from datetime import timedelta

        dt = parser.parse(datetime_str, fuzzy=True, default=datetime.now())
        dt = dt.replace(minute=0 if dt.minute < 30 else 30, second=0, microsecond=0)

        target_date = dt.date().isoformat()
        target_time = dt.time()

        start_dt = datetime.combine(dt.date(), target_time) - timedelta(hours=1)
        end_dt = datetime.combine(dt.date(), target_time) + timedelta(hours=1)
        start_time = start_dt.time().strftime("%H:%M:%S")
        end_time = end_dt.time().strftime("%H:%M:%S")

        result = (
            supabase.table('appointments')
            .select("appointment_time, status")
            .eq("appointment_date", target_date)
            .execute()
        )

        if not result.data:
            return "No appointments found for that date."

        conflicts = []
        for record in result.data:
            if start_time <= record["appointment_time"] <= end_time and record["status"] == "Booked":
                conflicts.append(record["appointment_time"])

        if conflicts:
            return f"⚠ Conflicts found at: {', '.join(conflicts)}"
        return "✓ No conflicts found."

    except Exception as e:
        return f"Error checking conflicts: {e}"

@tool
def cancel_appointment(appointment_id: str) -> str:
    """Cancel an existing appointment by ID."""
    try:
        result = (
            supabase.table('appointments')
            .update({"status": "Available", "student_id": None})
            .eq("appointment_id", appointment_id)
            .execute()
        )

        if result.data:
            return f"✓ Appointment {appointment_id} cancelled successfully."
        return f"✗ Appointment {appointment_id} not found."

    except Exception as e:
        return f"Error cancelling appointment: {e}"

@tool
def update_appointment(old_appointment_id: str, student_id: str) -> str:
    """Update an existing appointment to a new time by first canceling, then showing available slots."""
    try:
        cancel_result = cancel_appointment.invoke({"appointment_id": old_appointment_id})

        if "✓" not in cancel_result:
            return cancel_result

        slots_result = get_nearest_available_slot.invoke({})

        return f"Previous appointment cancelled.\n\n{slots_result}\n\nPlease confirm which slot you'd like to book."

    except Exception as e:
        return f"Error updating appointment: {e}"

@tool
def retrieve_treatment_info(condition: str, severity: str, k: int = 5) -> List[Document]:
    """
    Retrieve treatment plans and recommendations from the knowledge base
    for a specific mental health condition and severity level.
    """
    db = FAISS.load_local(
        "faiss_index",
        embeddings=OpenAIEmbeddings(),
        allow_dangerous_deserialization=True
    )
    query = f"treatment plan, advices or recommendations for {condition} at {severity} severity level"
    return db.similarity_search(query, k=k)

# Bind all tools to LLM
llm_with_tools_full = llm.bind_tools([
    retrieve_treatment_info,
    get_nearest_available_slot,
    book_appointment,
    check_conflicts,
    cancel_appointment,
    update_appointment
])
