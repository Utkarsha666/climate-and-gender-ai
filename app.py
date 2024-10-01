import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv
import pandas as pd
import seaborn as sns
import requests
from pymongo import MongoClient
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import GraphCypherQAChain
from langchain_groq import ChatGroq
from langchain_community.graphs import Neo4jGraph
from prophet import Prophet


# Load environment variables from .env file
load_dotenv()

# Access the variables
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
groq_api_key = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


nasa_url = "https://power.larc.nasa.gov/api/projection/daily/point?start=20200101&end=20240928&latitude=27.7103&longitude=85.3222&community=ag&parameters=PRECTOTCORR%2CT2M&format=json&user=utkarsha&header=true&time-standard=utc&model=ensemble&scenario=ssp126"

# Load the Gender Inequality Index (GII) data
url = "https://data.humdata.org/dataset/5a1ea18e-9177-4e37-b91f-5631961bdb6c/resource/4539296c-289c-48a2-b0dc-3fc8dcad1b77/download/gii_gender_inequality_index_value.csv"
gii_data = pd.read_csv(url)

args = {
  "iso2code": "NP"
  }

# load digital gender gap
digital_gender_gap = requests.get('http://digitalgendergaps.org/api/v1/query_specific_country?iso2code=NP', params = args)
digital_gender_gap = digital_gender_gap.json()
df_digital_gender_gap = pd.DataFrame(digital_gender_gap['data']['NP']).T
df_digital_gender_gap.index = pd.to_datetime(df_digital_gender_gap.index, format='%Y%m')

llm_groq = ChatGroq(groq_api_key=groq_api_key,model_name="Gemma-7b-It")
#######################################################################################################################

def get_climate_data(nasa_url):

    response = requests.get(nasa_url)

    if response.status_code == 200:
        data = response.json()
        parameters = data['properties']['parameter']
        dates = list(parameters['PRECTOTCORR'].keys())

        precipitation = [parameters['PRECTOTCORR'][date] for date in dates]
        temperature = [parameters['T2M'][date] for date in dates]
        
        climate_change_df = pd.DataFrame({
            'Date': dates,
            'Precipitation': precipitation,
            'Temperature': temperature
        })

        return climate_change_df

def forecast_temperature_and_precipitation(df):

    df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')

    precipitation_df = df[['Date', 'Precipitation']].rename(columns={'Date': 'ds', 'Precipitation': 'y'})
    precipitation_model = Prophet()
    precipitation_model.fit(precipitation_df)

    # forecasting
    future_precipitation = precipitation_model.make_future_dataframe(periods=365)
    forecast_precipitation = precipitation_model.predict(future_precipitation)

    temperature_df = df[['Date', 'Temperature']].rename(columns={'Date': 'ds', 'Temperature': 'y'})

    #model for Temperature
    temperature_model = Prophet()
    temperature_model.fit(temperature_df)

    # forecasting
    future_temperature = temperature_model.make_future_dataframe(periods=365)
    forecast_temperature = temperature_model.predict(future_temperature)

    # Plot for Precipitation Forecast
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(precipitation_df['ds'], precipitation_df['y'], label='Observed Precipitation', color='blue')
    ax1.plot(forecast_precipitation['ds'], forecast_precipitation['yhat'], label='Forecasted Precipitation', color='orange')
    ax1.fill_between(forecast_precipitation['ds'], 
                     forecast_precipitation['yhat_lower'], 
                     forecast_precipitation['yhat_upper'], 
                     color='orange', alpha=0.2, label='Confidence Interval')
    ax1.set_title('Precipitation Forecast')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Precipitation')
    ax1.legend()
    ax1.grid(True)

    # Plot for Temperature Forecast
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(temperature_df['ds'], temperature_df['y'], label='Observed Temperature', color='blue')
    ax2.plot(forecast_temperature['ds'], forecast_temperature['yhat'], label='Forecasted Temperature', color='red')
    ax2.fill_between(forecast_temperature['ds'], 
                     forecast_temperature['yhat_lower'], 
                     forecast_temperature['yhat_upper'], 
                     color='red', alpha=0.2, label='Confidence Interval')
    ax2.set_title('Temperature Forecast')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Temperature')
    ax2.legend()
    ax2.grid(True)

    return fig1, fig2


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
    
def plot_model_prediction(df, column_name, title):
    """
    Function to plot a line chart using Seaborn for the given column in the DataFrame
    in a Streamlit app.

    Parameters:
    - df: DataFrame containing the data
    - column_name: The column to be plotted
    - title: The title of the plot
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Seaborn lineplot
    sns.lineplot(data=df, x=df.index, y=column_name, ax=ax)
    
    # Set plot title and remove top and right spines
    ax.set_title(title)
    ax.spines[['top', 'right']].set_visible(False)
    
    # Display the plot in Streamlit
    st.pyplot(fig)

##################################################################################################################

def connect_mongoDB():
    MONGO_USER = os.getenv("MONGO_USER")
    MONGO_PASSWORD = os.getenv("MONGO_PASSWORD")

    # MongoDB Atlas connection
    uri = f'mongodb+srv://{MONGO_USER}:{MONGO_PASSWORD}@cluster0.5sbsz.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0'
    client = MongoClient(uri)

    # Select the database and collection
    db = client['CCandGender']  
    collection = db['embeddings']    
    return collection

def get_vector_retriever(collection):
    # Define the Atlas Vector Search Index name
    ATLAS_VECTOR_SEARCH_INDEX_NAME = 'vector_index'
    embedding_model = GoogleGenerativeAIEmbeddings(model='models/embedding-001')

    # Define your vector search engine using MongoDB Atlas
    vector_search = MongoDBAtlasVectorSearch.from_documents(
        documents=[],  # no need to pass documents here if they are already in MongoDB
        embedding=embedding_model,
        collection=collection,
        index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME
    )
    retriever = vector_search.as_retriever(search_type='similarity', search_kwargs={'k': 3})

    return retriever

def question(user_input):
    if user_input: 

        model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.2)

        template = """
            Task: Answer the question using only the provided context: {context}.

            Context: The documents consist of reports, policies, and publications related to Climate Change and Gender Inequality.

            Instructions:

                Provide accurate, detailed answers strictly based on the given documents.
                Cite relevant references on the relationship between gender and climate change.
                Do not introduce information beyond the documents or make assumptions.
                If the query is unclear or lacks sufficient detail, ask for clarification before responding.
                Maintain a neutral tone in your answer.
                Avoid starting with phrases like "The provided text..."

            Question: {question}
        """


        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template=template
        )

        database = connect_mongoDB()
        retriever = get_vector_retriever(database)

        rag_chain = (
        {
            "context": retriever,
            "question": RunnablePassthrough()
        }
        | prompt_template
        | model
        | StrOutputParser()
        ) 
            
        answer = rag_chain.invoke(user_input)
        return answer


# Set the sidebar title
st.sidebar.title("Climate and Gender AI")
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

# Create a dropdown menu in the sidebar
relation_option = st.sidebar.selectbox(
    'Select relation to visualize entities:',
    ('Impact','Effects', 'Contributes', 'CAUSES', 'Promotes', 'Effected-By', 'Applies-To', 'Funds', 'Developed', 'Damages', 'Governing', 'Specifies', 'Promotes')
)

# Create a dropdown menu in the sidebar
digital_gender_gap_option = st.sidebar.selectbox(
    'Select digital gender gap',
    ('Internet Online','Internet Offline', 'Internet Both', 'Mobile Online', 'Mobile Offline', 'Mobile Both', 'Mobile GSMA')
)

climate_change_df = get_climate_data(nasa_url)
user_input = st.sidebar.text_area("Ask question to Bot: Climate Change and Gender (Reports, Policies, assesment etc Nepal)")
submit_button = st.sidebar.button("ASK ME")

# Display the submitted text
if submit_button:
    answer = question(user_input)
    st.sidebar.write(answer)

#########################################################################################################################

if __name__ == '__main__':
    st.title("Gender and Climate Change")

    st.subheader("Relationship visualizer")
    relation_queries = {
        'Effects': {
            'query': "MATCH p=()-[:AFFECTS]->() RETURN p LIMIT 25;",
            'description': "Edges: Effect Relationship<br>Nodes: Entities"
        },
        'CAUSES': {
            'query': "MATCH p=()-[:CAUSES]->() RETURN p LIMIT 25;",
            'description': "Edges: Causes Relationship<br>Nodes: Entities"
        },
        'Integrates': {
            'query': "MATCH p=()-[:INTEGRATES]->() RETURN p LIMIT 25;",
            'description': "Edges: Integrates Relationship<br>Nodes: Entities"
        },
        'Damages': {
            'query': "MATCH p=()-[:DAMAGES]->() RETURN p LIMIT 25;",
            'description': "Edges: Damages Relationship<br>Nodes: Entities"
        },
        'Effected-By': {
            'query': "MATCH p=()-[:AFFECTED_BY]->() RETURN p LIMIT 25;",
            'description': "Edges: Effected By Relationship<br>Nodes: Entities"
        },
        'Impact': {
            'query': "MATCH p=()-[:IMPACTS]->() RETURN p LIMIT 25;",
            'description': "Edges: Impacts Relationship<br>Nodes: Entities"
        },
        'Applies-To': {
            'query': "MATCH p=()-[:APPLIES_TO]->() RETURN p LIMIT 25;",
            'description': "Edges: Applies to Relationship<br>Nodes: Entities"
        },
        'Funds': {
            'query': "MATCH p=()-[:FUNDS]->() RETURN p LIMIT 25;",
            'description': "Edges: Funds Relationship<br>Nodes: Entities"
        },
        'Contributes': {
            'query': "MATCH p=()-[:CONTRIBUTES_TO]->() RETURN p LIMIT 25;",
            'description': "Edges: Contributes Relationship<br>Nodes: Entities"
        },
        'Developed': {
            'query': "MATCH p=()-[:DEVELOPED]->() RETURN p LIMIT 25;",
            'description': "Edges: Developed Relationship<br>Nodes: Entities"
        },
        'Governing': {
            'query': "MATCH p=()-[:GOVERNING]->() RETURN p LIMIT 25;",
            'description': "Edges: Governing Relationship<br>Nodes: Entities"
        },
        'Specifies': {
            'query': "MATCH p=()-[:SPECIFIES]->() RETURN p LIMIT 25;",
            'description': "Edges: Specifies Relationship<br>Nodes: Entities"
        },
        'Promotes': {
            'query': "MATCH p=()-[:PROMOTES]->() RETURN p LIMIT 25;",
            'description': "Edges: Promotes Relationship<br>Nodes: Entities"
        }
    }

    if relation_option in relation_queries:
        query_info = relation_queries[relation_option]
        query = query_info['query']
        st.markdown(query_info['description'], unsafe_allow_html=True)


    # Run the selected query
    neo4j_paths = run_cypher_query(query, driver)

    # Convert to NetworkX graph
    G = neo4j_to_networkx(neo4j_paths)

    # Plot the graph in the Streamlit app
    plot_graph(G)

    # Button to explain the graph in natural language
    if st.sidebar.button("Explain Graph"):
        entities_info = []
        for path in neo4j_paths:
            for rel in path.relationships:
                start_node = rel.start_node
                end_node = rel.end_node
                relationship = rel
                entities_info.append(f"Entity: {start_node['id']} Relationship: {relationship.type} Entity: {end_node['id']}")

        entities_info_str = "\n".join(entities_info)
        prompt = f"Explain the relationships between the following entities in Neo4j:\n{entities_info_str}\n in 20 words for each realtion"

        # Define the messages for the chat
        messages = [
            {"role": "system", "content": "You are a expert assistant whose job is to find relation between entities"},
            {"role": "user", "content": prompt}
        ]

        response = llm_groq.invoke(messages)
        st.sidebar.write(response.content)
    ############################################################################################################################
    fig_precipitation, fig_temperature = forecast_temperature_and_precipitation(climate_change_df)
    col1, col2 = st.columns(2)

    with col1:
        st.write("Precipitation Forecast")
        st.pyplot(fig_precipitation)

    with col2:
        st.write("Temperature Forecast")
        st.pyplot(fig_temperature)


    # Filter and plot the time series data for Nepal (Gender Inequality Index)
    nepal_data = gii_data[gii_data["country"] == "Nepal"]
    nepal_data_grouped = nepal_data.groupby("year")["value"].mean().reset_index()

    col1, col2 = st.columns([1, 2])  
    with col1:
        st.write("Gender Inequality Index Time Series (Nepal)")
        plot_time_series(nepal_data_grouped, "Gender Inequality Index")
    
    with col2:
        st.write("Digital Gender Gap")
        if digital_gender_gap_option == 'Internet Online':
            plot_model_prediction(df_digital_gender_gap, 'internet_online_model_prediction', 'Internet Online')
        elif digital_gender_gap_option == 'Internet Offline':
            plot_model_prediction(df_digital_gender_gap, 'internet_offline_model_prediction', 'Internet Offline')
        elif digital_gender_gap_option == 'Internet Both':
            plot_model_prediction(df_digital_gender_gap, 'internet_online_offline_model_prediction', 'Internet Both')
        elif digital_gender_gap_option == 'Mobile Online':
            plot_model_prediction(df_digital_gender_gap, 'mobile_online_model_prediction', 'Mobile Online')
        elif digital_gender_gap_option == 'Mobile Offline':
            plot_model_prediction(df_digital_gender_gap, 'mobile_offline_model_prediction', 'Mobile Offline')
        elif digital_gender_gap_option == 'Mobile Both':
            plot_model_prediction(df_digital_gender_gap, 'mobile_online_offline_model_prediction', 'Mobile Both')
        elif digital_gender_gap_option == 'Mobile GSMA':
            plot_model_prediction(df_digital_gender_gap, 'ground_truth_mobile_gg', 'Mobile GSMA')





