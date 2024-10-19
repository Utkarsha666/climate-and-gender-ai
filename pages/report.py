import streamlit as st
import agents.interview
import agents.analyst
from agents.research import research_graph_builder

st.set_page_config(page_title="Report Generation")

# User inputs
max_analysts = st.number_input("Max Number of Analysts", min_value=1, max_value=10, value=2)
topics = [
    "The Intersection of Climate Change and Gender Equality in Nepal",
    "Climate Changes and its Impact in Nepal",
    "Climate Change Policy in Nepal",
    "Gender Equality in Nepal",
    "Report of Climate Related Disaster in Nepal",
    "Gender Based Agriculture in Nepal",
    "Women Effected by Climate Change in Nepal",
    "Climate Related Disaster Assesment in Nepal",
    "Climate Related Disaster Prone Areas of Nepal",
    "Effects of Climate Change in Nepal",
    "Early Warning System in Nepal",
    "Greenhouse Gas Emmision in Nepal",
    "Healthcare, Climate Change and Gender Equity Policy in Nepal"
]

topic = st.sidebar.selectbox("Choose your topic", topics)
thread = {"configurable": {"thread_id": "1"}}

graph = research_graph_builder()

# Run the graph until the first interruption
for event in graph.stream({"topic": topic,
                        "max_analysts": max_analysts},
                        thread,
                        stream_mode="values"):

    analysts = event.get('analysts', '')
    if analysts:
        for analyst in analysts:
            print(f"**Name:** {analyst.name}")
            print(f"**Affiliation:** {analyst.affiliation}")
            print(f"**Role:** {analyst.role}")
            print(f"**Description:** {analyst.description}")
            print("---")

feedback = st.sidebar.text_input("Enter feedback for the analysts (e.g., add a specific analyst):",  value="Add in the Country Director of NGO of Nepal")

if st.sidebar.button("Submit Feedback"):
    # Update the state with user feedback
    graph.update_state(thread, {"human_analyst_feedback": feedback}, as_node="human_feedback")

    # Display a confirmation message
    st.sidebar.success("Feedback submitted successfully!")

    # Check
    for event in graph.stream(None, thread, stream_mode="values"):
        analysts = event.get('analysts', '')
        if analysts:
            for analyst in analysts:
                st.sidebar.write(f"**Name:** {analyst.name}")
                st.sidebar.write(f"**Affiliation:** {analyst.affiliation}")
                st.sidebar.write(f"**Role:** {analyst.role}")
                st.sidebar.write(f"**Description:** {analyst.description}")
                st.sidebar.write("---")


    # Confirm we are done
    graph.update_state(thread, {"human_analyst_feedback":
                                None}, as_node="human_feedback")

    # Continue updates
    with st.spinner("researching and generating report..."):
        for event in graph.stream(None, thread, stream_mode="updates"):
            st.write("--Node--")
            node_name = next(iter(event.keys()))
            st.write(node_name)

    # Final report
    final_state = graph.get_state(thread)
    report = final_state.values.get('final_report')
    st.subheader("Final Report")
    st.write(report)
