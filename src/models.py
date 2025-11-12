from pydantic import BaseModel, Field
from typing import Annotated, Literal, List, Optional
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

## UNIFIED STATE - Combines all 3 graphs
## UNIFIED STATE - Combines all 3 graphs
class UnifiedState(TypedDict):
    """
    Combined state for conversation, questionnaire, and recommendation workflows.
    """
    # From Graph 1 (Conversation)
    messages: Annotated[List, add_messages]
    session_id: str
    iterator: int
    rag_context: Optional[str]

    # From Graph 2 (Questionnaire)
    disorder: str
    question_id: list
    current_question_id: str
    answer: list
    user_answer: str
    score: int
    reword_questionnaire: dict
    next_node: str
    total_score: int
    score_label: str

    # From Graph 3 (Recommendation)
    student_id: str
    condition: Optional[str]
    severity: Optional[str]
    route: Optional[Literal["treatment_plan", "appointment"]]
    recommendation: Optional[str]
    appointment_confirmed: Optional[bool]
    suggested_appointment_id: Optional[str]
    conversation_history: List[dict]
    user_message: Optional[str]
    workflow_stage: Optional[str]


class Feedback(BaseModel):
    condition: Literal["anxiety", "depression", "stress"] = Field(
        description="Decide from the chat history if the user has condition of anxiety or depression or stress.",
    )
    reasoning: str = Field(
        description="Give a reasoning of the chosen condition and why you chose it",
    )

