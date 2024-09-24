import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv
import pandas as pd
import seaborn as sns

# Load environment variables from .env file
load_dotenv()

# Access the variables
uri = os.getenv("NEO4J_URI")
user = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")
groq_api_key = os.getenv("GROQ_API_KEY")

# Load the Gender Inequality Index (GII) data
# Load the Gender Inequality Index data
url = "https://data.humdata.org/dataset/5a1ea18e-9177-4e37-b91f-5631961bdb6c/resource/4539296c-289c-48a2-b0dc-3fc8dcad1b77/download/gii_gender_inequality_index_value.csv"
gii_data = pd.read_csv(url)


#######################################################################################################################

def run_cypher_query(query, driver):
    with driver.session() as session:
        result = session.run(query)
        return [record["p"] for record in result]

def neo4j_to_networkx(neo4j_paths):
    G = nx.Graph()
    
    for path in neo4j_paths:
        nodes = path.nodes
        relationships = path.relationships
        for relationship in relationships:
            G.add_edge(relationship.start_node["id"], relationship.end_node["id"], type=relationship.type)

    return G

def plot_graph(G):
    plt.figure(figsize=(8, 6))
    nx.draw(G, with_labels=True, node_color="lightblue", edge_color="gray", node_size=500, font_size=10)
    st.pyplot(plt)  


def plot_time_series(data, string):
    # Set a more modern style using seaborn
    sns.set(style="whitegrid")
    
    # Adjust the figure size to make it smaller (e.g., 6x4 inches)
    plt.figure(figsize=(6, 3))  
    
    # Plot with improved aesthetics
    plt.plot(data["year"], data["value"], marker='o', linestyle='-', color='royalblue', linewidth=2, markersize=6)
    
    # Adding labels for each data point
    for i, row in data.iterrows():
        plt.text(row["year"], row["value"], f'{row["value"]:.2f}', color="black", ha="right", fontsize=8)
    
    # Add title and customize font size
    plt.title(string, fontsize=12, fontweight='bold')
    
    # Adjust the limits for better visualization
    plt.ylim(data["value"].min() - 0.02, data["value"].max() + 0.02)
    plt.xlim(data["year"].min() - 1, data["year"].max() + 1)
    
   
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
    
    
    st.pyplot(plt)



###################################################################################################################

if __name__ == '__main__':

    st.title("Gender Inequality Index")

    # Filter and plot the time series data for Nepal (Gender Inequality Index)
    nepal_data = gii_data[gii_data["country"] == "Nepal"]
    nepal_data_grouped = nepal_data.groupby("year")["value"].mean().reset_index()

    st.subheader("Gender Inequality Index Time Series (Nepal)")
    plot_time_series(nepal_data_grouped, "Gender Inequality Index")
    st.title("Relationship Visualizer")

    # Create a dropdown menu in the sidebar
    option = st.sidebar.selectbox(
        'Select a query to visualize relationships:',
        ('Applies-To','Effects', 'CAUSES', 'Effected-By', 'Impact', 'Funds')
    )

    driver = GraphDatabase.driver(uri, auth=(user, password))

    # Define different Cypher queries based on the dropdown selection
    if option == 'Effects':
        query = "MATCH p=()-[:AFFECTS]->() RETURN p LIMIT 25;"
    elif option == 'CAUSES':
        query = "MATCH p=()-[:CAUSES]->() RETURN p LIMIT 25;"
    elif option == 'Effected-By':
        query = "MATCH p=()-[:AFFECTED_BY]->() RETURN p LIMIT 25;"
    elif option == 'Impact':
        query = "MATCH p=()-[:IMPACTS]->() RETURN p LIMIT 25;"
    elif option == 'Applies-To':
        query = "MATCH p=()-[:APPLIES_TO]->() RETURN p LIMIT 25;"
    elif option == 'Funds':
        query = "MATCH p=()-[:FUNDS]->() RETURN p LIMIT 25;"

    # Run the selected query
    neo4j_paths = run_cypher_query(query, driver)

    # Convert to NetworkX graph
    G = neo4j_to_networkx(neo4j_paths)

    # Plot the graph in the Streamlit app
    plot_graph(G)

    # Close driver connection
    driver.close()
