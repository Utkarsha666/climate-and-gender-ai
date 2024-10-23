import streamlit as st
import agents.interview
import agents.analyst
from agents.research import research_graph_builder


st.set_page_config(page_title="Report Generation")

max_analysts = st.sidebar.number_input("Max Number of Analysts", min_value=1, max_value=10, value=1)
topics = [
    "The Intersection of Climate Change and Gender Equality in Nepal",
    "Climate Changes and its Impact in Nepal",
    "Climate Change Policy in Nepal",
    "Gender Equality in Nepal",
    "Report of Climate Related Disaster in Nepal",
    "Gender Based Agriculture in Nepal",
    "Women Effected by Climate Change in Nepal",
    "Climate Related Disaster in Nepal (Flood, Lanslide)",
    "Climate Related Disaster Prone Areas of Nepal",
    "Effects of Climate Change in Nepal",
    "Early Warning System in Nepal",
    "Greenhouse Gas Emmision in Nepal",
    "Healthcare, Climate Change and Gender Equity Policy in Nepal"
]

topic = st.sidebar.selectbox("Choose your topic", topics)
thread = {"configurable": {"thread_id": "1"}}

graph = research_graph_builder()

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
    st.sidebar.info("Generating report may fail due to token limits. If it does retry after 1 minute to resolve the issue.")
    graph.update_state(thread, {"human_analyst_feedback": feedback}, as_node="human_feedback")
    st.sidebar.success("Feedback submitted successfully!")
    for event in graph.stream(None, thread, stream_mode="values"):
        analysts = event.get('analysts', '')
        if analysts:
            for analyst in analysts:
                st.sidebar.write(f"**Name:** {analyst.name}")
                st.sidebar.write(f"**Affiliation:** {analyst.affiliation}")
                st.sidebar.write(f"**Role:** {analyst.role}")
                st.sidebar.write(f"**Description:** {analyst.description}")
                st.sidebar.write("---")
    graph.update_state(thread, {"human_analyst_feedback":
                                None}, as_node="human_feedback")
    with st.spinner("researching and generating report..."):
        for event in graph.stream(None, thread, stream_mode="updates"):
            st.write("--Node--")
            node_name = next(iter(event.keys()))
            st.write(node_name)
    final_state = graph.get_state(thread)
    report = final_state.values.get('final_report')
    st.subheader("Final Report")
    st.write(report)
