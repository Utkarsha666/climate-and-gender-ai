from typing import List
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from IPython.display import Image, display, FileLink
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq

llm=ChatGroq(temperature=0.2, model_name="mixtral-8x7b-32768",
             api_key='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')

# Define the Analyst model
class Analyst(BaseModel):
    affiliation: str = Field(description="Primary affiliation of the analyst.")
    name: str = Field(description="Name of the analyst.")
    role: str = Field(description="Role of the analyst in the context of the topic.")
    description: str = Field(description="Description of the analyst focus, concerns, and motives.")
    
    @property
    def persona(self) -> str:
        return f"Name: {self.name}\nRole: {self.role}\nAffiliation: {self.affiliation}\nDescription: {self.description}\n"

# Define the Perspectives model
class Perspectives(BaseModel):
    analysts: List[Analyst] = Field(description="Comprehensive list of analysts with their roles and affiliations.")

# Define the state type
class GenerateAnalystsState(TypedDict):
    topic: str  # Research topic
    max_analysts: int  # Number of analysts
    human_analyst_feedback: str  # Human feedback
    analysts: List[Analyst]  # Analyst asking questions

# Define instructions for the analyst
analyst_instructions = """You are tasked with creating a set of AI analyst personas. Follow these instructions carefully:

1. First, review the research topic:
{topic}

2. Examine any editorial feedback that has been optionally provided to guide creation of the analysts:

{human_analyst_feedback}

3. Determine the most interesting themes based upon documents and/or feedback above.

4. Pick the top {max_analysts} themes.

5. Assign one analyst to each theme."""

# Define the function to create analysts
def create_analysts(state: GenerateAnalystsState):
    topic = state['topic']
    max_analysts = state['max_analysts']
    human_analyst_feedback = state.get('human_analyst_feedback', '')

    # Placeholder for the LLM invocation
    structured_llm = llm.with_structured_output(Perspectives)

    # System message
    system_message = analyst_instructions.format(
        topic=topic,
        human_analyst_feedback=human_analyst_feedback,
        max_analysts=max_analysts
    )

    # Generate analysts
    analysts = structured_llm.invoke([SystemMessage(content=system_message)] + 
                                      [HumanMessage(content="Generate the set of analysts.")])

    return {"analysts": analysts.analysts}

# Define a no-op human feedback function
def human_feedback(state: GenerateAnalystsState):
    pass

# Define the function to decide the next node
def should_continue(state: GenerateAnalystsState):
    human_analyst_feedback = state.get('human_analyst_feedback', None)
    if human_analyst_feedback:
        return "create_analysts"
    return END

def build_analyst_state_graph():
    # Build the state graph
    builder = StateGraph(GenerateAnalystsState)
    builder.add_node("create_analysts", create_analysts)
    builder.add_node("human_feedback", human_feedback)
    builder.add_edge(START, "create_analysts")
    builder.add_edge("create_analysts", "human_feedback")
    builder.add_conditional_edges("human_feedback", should_continue, ["create_analysts", END])

    # Compile the graph
    memory = MemorySaver()
    graph = builder.compile(interrupt_before=['human_feedback'], checkpointer=memory)

    return graph

   
