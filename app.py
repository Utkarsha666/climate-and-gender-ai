import streamlit as st
import networkx as nx
from neo4j import GraphDatabase
import numpy as np
import os
from dotenv import load_dotenv
import pandas as pd
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
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from prophet import Prophet
import altair as alt
from datetime import timedelta, date
import plotly.graph_objects as go
from sklearn.preprocessing import QuantileTransformer
import plotly.express as px
import joblib

st.set_page_config(
    page_title="Dashboard",
)

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
groq_api_key = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Nasa Temperature and Percipiration 
nasa_url = "https://power.larc.nasa.gov/api/projection/daily/point?start=20200101&end=20241018&latitude=27.7103&longitude=85.3222&community=ag&parameters=PRECTOTCORR%2CT2M&format=json&user=utkarsha&header=true&time-standard=utc&model=ensemble&scenario=ssp126"
model_temp = joblib.load('models/xgboost_temp_model.pkl')
model_precip = joblib.load('models/xgboost_precip_model.pkl')
qt = QuantileTransformer(output_distribution='normal')

# Load the Gender Inequality Index (GII) data
gii_url = "https://data.humdata.org/dataset/5a1ea18e-9177-4e37-b91f-5631961bdb6c/resource/4539296c-289c-48a2-b0dc-3fc8dcad1b77/download/gii_gender_inequality_index_value.csv"
gii_model = Prophet()

# gender gap ratio 
ggr_df = 'datasets/WEF-GGR.xlsx'

args = {
  "iso2code": "NP"
  }

# load digital gender gap
digital_gender_gap = requests.get('http://digitalgendergaps.org/api/v1/query_specific_country?iso2code=NP', params = args)
digital_gender_gap = digital_gender_gap.json()
df_digital_gender_gap = pd.DataFrame(digital_gender_gap['data']['NP']).T
df_digital_gender_gap.index = pd.to_datetime(df_digital_gender_gap.index, format='%Y%m')

# small model for simple exlaining the graph
llm_groq = ChatGroq(groq_api_key=groq_api_key,model_name="Gemma-7b-It")

#######################################################################################################################

def load_and_prepare_ggr(ggr_df):
    data = pd.read_excel(ggr_df)
    nepal_data = data[data['Economy Name'] == 'Nepal']
    nepal_data.set_index('Indicator', inplace=True)
    nepal_data = nepal_data.transpose()
    nepal_data = nepal_data.drop(['Economy ISO3', 'Economy Name', 'Indicator ID', 'Attribute 1', 'Attribute 2', 'Attribute 3', 'Partner'], errors='ignore')
    nepal_data.index = pd.to_datetime(nepal_data.index, errors='coerce').dropna()
    indicators = nepal_data.columns.tolist()
    indicators = pd.Series(indicators).duplicated().cumsum().astype(str).radd(indicators).tolist()
    nepal_data.columns = indicators
    default_indicators = indicators[:2] if len(indicators) >= 2 else indicators
    return nepal_data, indicators, default_indicators

def load_and_prepare_gii(gii_url):
    gii_data = pd.read_csv(gii_url)
    return gii_data

def load_climate_change_indicator():
    climate_related_disaster_df = pd.read_csv('datasets/CSV/Nepal_NP_All_Indicators/14_Climate-related_Disasters_Frequency.csv')
    climate_driven_inform_risk_df = pd.read_csv('datasets/CSV/Nepal_NP_All_Indicators/15_Climate-driven_INFORM_Risk.csv')
    fossil_fuel_subsidies_df = pd.read_csv('datasets/CSV/Nepal_NP_All_Indicators/09_Fossil_Fuel_Subsidies.csv')
    environmental_protection_expenditures_df = pd.read_csv('datasets/CSV/Nepal_NP_All_Indicators/08_Environmental_Protection_Expenditures.csv')
    forest_and_carbon_df = pd.read_csv('datasets/CSV/Nepal_NP_All_Indicators/13_Forest_and_Carbon.csv')
    return climate_related_disaster_df, climate_driven_inform_risk_df, fossil_fuel_subsidies_df, environmental_protection_expenditures_df, forest_and_carbon_df

def load_gender_statistics():
    # Load Gender Statistics ( Source : World Bank )
    df = pd.read_excel("datasets/P_Data_Extract_From_Gender_Statistics.xlsx")
    df = df.drop(columns=['Series Code', 'Country Name', 'Country Code'])
    df = df.groupby('Series Name').apply(lambda x: x.sort_values(by='Series Name')).reset_index(drop=True)
    data_long = pd.melt(df, id_vars=['Series Name'], var_name='Year', value_name='Value')
    data_long['Year'] = data_long['Year'].str.extract('(\d{4})').astype(int)
    data_long['Value'] = pd.to_numeric(data_long['Value'], errors='coerce')
    data_long = data_long.dropna(subset=['Value'])
    series_names = data_long['Series Name'].unique()
    return data_long, series_names

def plot_gender_statistics(data_long, selected_series):
    if selected_series:
        filtered_data = data_long[data_long['Series Name'].isin(selected_series)]
        fig = px.line(filtered_data, x='Year', y='Value', color='Series Name', title='Gender Statistics')

        fig.update_layout(
            xaxis_title='Year',
            yaxis_title='Value',
            width=1200,  
            showlegend=False,
            margin=dict(l=20, r=20, t=30, b=20)
        )

        st.plotly_chart(fig)

def plot_ggr(nepal_data, selected_indicators):
    if selected_indicators:
        filtered_data = nepal_data[selected_indicators].reset_index()
        filtered_data = filtered_data.melt(id_vars='index', var_name='Indicator', value_name='Value')
        filtered_data.rename(columns={'index': 'Year'}, inplace=True)

        filtered_data = filtered_data.groupby(['Year', 'Indicator']).mean().reset_index()

        fig = px.line(filtered_data, x='Year', y='Value', color='Indicator', title='Gender Gap Report for Nepal')

        fig.update_layout(
            xaxis_title='Year',
            yaxis_title='Value',
            width=1000,  
            showlegend=False,
            margin=dict(l=20, r=20, t=30, b=20)
        )

        st.plotly_chart(fig)

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
        climate_change_df['Date'] = pd.to_datetime(climate_change_df['Date'])
        climate_change_df.set_index('Date', inplace=True)
        return climate_change_df
    
def forecast_temperature_and_precipitation(climate_change_df):
    qt = QuantileTransformer(output_distribution='normal')
    climate_change_df['Transformed_Precipitation'] = qt.fit_transform(climate_change_df[['Precipitation']])

    # Generate dates for the next 10 days
    last_date = climate_change_df.index[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, 11)]

    # Create a DataFrame for the future dates
    future_df = pd.DataFrame(index=future_dates)
    future_df['month'] = future_df.index.month
    future_df['dayofyear'] = future_df.index.dayofyear
    future_df['dayofmonth'] = future_df.index.day
    future_df['dayofweek'] = future_df.index.dayofweek

    # Add lag features from the last available data
    for lag in range(1, 4):
        future_df[f'temp_lag_{lag}'] = climate_change_df['Temperature'].shift(lag).iloc[-1]
        future_df[f'precip_lag_{lag}'] = climate_change_df['Precipitation'].shift(lag).iloc[-1]

    # Add rolling mean and other features
    future_df['temp_roll_mean'] = climate_change_df['Temperature'].rolling(window=7).mean().iloc[-1]
    future_df['precip_roll_mean'] = climate_change_df['Precipitation'].rolling(window=7).mean().iloc[-1]
    future_df['precip_diff'] = climate_change_df['Precipitation'].diff().iloc[-1]
    future_df['precip_pct_change'] = climate_change_df['Precipitation'].pct_change().iloc[-1]

    # Ensure the order of columns in future_df matches the training data
    feature_names = model_temp.get_booster().feature_names
    future_df = future_df[feature_names]

    # Predict temperature
    future_temp_predictions = model_temp.predict(future_df)

    # Predict precipitation and reverse the quantile transformation
    future_precip_predictions = model_precip.predict(future_df)
    future_precip_predictions = qt.inverse_transform(future_precip_predictions.reshape(-1, 1)).flatten()

    # Create a DataFrame for the predictions
    future_predictions_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted_Temperature': future_temp_predictions,
        'Predicted_Precipitation': future_precip_predictions
    })

    # Plot the predictions
    st.line_chart(future_predictions_df.set_index('Date')[['Predicted_Temperature', 'Predicted_Precipitation']])


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
    pos = nx.spring_layout(G, seed=42)
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=0.5, color='#888'),
        hoverinfo='none',
        mode='lines')

    node_x = []
    node_y = []
    node_text = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(str(node))  

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        text=node_text,  
        textposition="middle center",  
        hoverinfo='text',
        marker=dict(
            showscale=True,
            colorscale='YlGnBu',
            size=20,  
            colorbar=dict(
                thickness=15,
                title='Node Connections',
                xanchor='left',
                titleside='right'
            ),
            line_width=2))

    fig = go.Figure(data=[edge_trace, node_trace],
                    layout=go.Layout(
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=0, l=0, r=0, t=0),
                        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),  
                        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))  

    st.plotly_chart(fig)


def forecast_gii(nepal_data):
    nepal_data = nepal_data[nepal_data["country"] == "Nepal"]
    nepal_data = nepal_data.groupby("year")["value"].mean()

    nepal_data_prophet = pd.DataFrame({
        'ds': pd.to_datetime(nepal_data.index, format='%Y'),  
        'y': nepal_data.values  
    })

    model = Prophet()
    model.fit(nepal_data_prophet)

    future = model.make_future_dataframe(periods=8, freq="AS")  
    forecast = model.predict(future)

    actual_data = nepal_data_prophet.copy()
    actual_data['type'] = 'Actual'
    forecast_data = forecast[['ds', 'yhat']].copy()
    forecast_data.rename(columns={'yhat': 'y'}, inplace=True)
    forecast_data['type'] = 'Forecast'

    combined_data = pd.concat([actual_data, forecast_data])
    actual_chart = alt.Chart(combined_data[combined_data['type'] == 'Actual']).mark_line(color='blue').encode(
        x='ds:T',
        y='y:Q',
        tooltip=['ds:T', 'y:Q']
    ).properties(title='Gender Inequality Index of Nepal Forecast')

    forecast_chart = alt.Chart(combined_data[combined_data['type'] == 'Forecast']).mark_line(color='red', strokeDash=[5, 5]).encode(
        x='ds:T',
        y='y:Q',
        tooltip=['ds:T', 'y:Q']
    )

    st.altair_chart(actual_chart + forecast_chart, use_container_width=True)
    
def plot_model_prediction(df, column_name, title):
    """
    Function to plot a line chart using Streamlit for the given column in the DataFrame.

    Parameters:
    - df: DataFrame containing the data
    - column_name: The column to be plotted
    - title: The title of the plot
    """
    chart_data = df[[column_name]].reset_index()
    chart = alt.Chart(chart_data.rename(columns={'index': 'Date'})).mark_line().encode(
        x='Date:T',
        y=column_name
    ).properties(
        title=title
    )

    st.altair_chart(chart, use_container_width=True)

def plot_climate_change_indicator(df, title, ylabel):
    indicators = df['Indicator'].unique()
    plot_data = pd.DataFrame()

    for indicator in indicators:
        indicator_data = df[df['Indicator'] == indicator]
        if title == 'Climate Related Disasters':
            years = [int(col) for col in indicator_data.columns[5:]]  
            values = indicator_data.iloc[0, 5:].to_numpy()  
        else:
            years = [int(col) for col in indicator_data.columns[9:]]  
            values = indicator_data.iloc[0, 9:].to_numpy()  

        temp_df = pd.DataFrame({
            'Year': years,
            'Value': values,
            'Indicator': indicator
        })
        plot_data = pd.concat([plot_data, temp_df])

    # Select the first four indicators by default
    default_indicators = indicators[:4] if len(indicators) >= 4 else indicators

    selected_indicators = st.multiselect('Select Indicators', indicators, default=default_indicators)

    filtered_data = plot_data[plot_data['Indicator'].isin(selected_indicators)]

    fig = px.line(filtered_data, x='Year', y='Value', color='Indicator', title=title, labels={'Value': ylabel})

    fig.update_layout(
        xaxis_title='Year',
        yaxis_title=ylabel,
        width=1000,  
        showlegend=False,
        margin=dict(l=20, r=20, t=30, b=20)
    )

    st.plotly_chart(fig)

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
    ATLAS_VECTOR_SEARCH_INDEX_NAME = 'vector_index'
    embedding_model = GoogleGenerativeAIEmbeddings(model='models/embedding-001')

    vector_search = MongoDBAtlasVectorSearch.from_documents(
        documents=[],  
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
    
    
##################################################################################################################################
##################################################################################################################################
##################################################################################################################################
nepal_data_grouped = load_and_prepare_gii(gii_url)
gender_statistics_data, series_names = load_gender_statistics()
(climate_related_disaster_df, climate_driven_inform_risk_df, fossil_fuel_subsidies_df, environmental_protection_expenditures_df, forest_and_carbon_df) = load_climate_change_indicator()
cci_df_dataframes = {
    'Climate Related Disasters': climate_related_disaster_df,
    'Climate Driven Inform Risk': climate_driven_inform_risk_df,
    'Fossil Fuel Subsidies': fossil_fuel_subsidies_df,
    'Environmental Protection Expenditures': environmental_protection_expenditures_df,
    'Forest and Carbon': forest_and_carbon_df
}

# Set the sidebar title
st.sidebar.title("Climate and Gender AI")
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

# Create a dropdown menu in the sidebar
relation_option = st.sidebar.selectbox(
    'Select relation to visualize entities:',
    ('Impact','Effects', 'Contributes', 'CAUSES', 'Promotes', 'Effected-By', 'Applies-To', 'Funds', 'Developed', 'Damages', 'Governing', 'Specifies', 'Promotes')
)

climate_change_indicator_option = st.sidebar.selectbox('Climate Change Indicator Nepal', list(cci_df_dataframes.keys()))
cci_df = cci_df_dataframes[climate_change_indicator_option]

climate_change_df = get_climate_data(nasa_url)

ggr_data, ggr_indicator, ggr_default_indicator = load_and_prepare_ggr(ggr_df)
selected_ggr_indicators = st.sidebar.multiselect('Select Gender Gap Indicators', ggr_indicator, default=ggr_default_indicator)

#Gender statistics
selected_gender_statistics_indicator = st.sidebar.multiselect('Select Gender Statistics', series_names, default=series_names[:2])


user_input = st.sidebar.text_area("Ask question to Bot: Climate Change and Gender (Reports, Policies, assesment etc Nepal)")
submit_button = st.sidebar.button("ASK ME")
if submit_button:
    answer = question(user_input)
    st.sidebar.write(answer)

#########################################################################################################################

if __name__ == '__main__':
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
    st.write("10 Days Climate Change Prediction")
    forecast_temperature_and_precipitation(climate_change_df)

    col1, col2 = st.columns([1, 1])  
    with col1:
        #st.write("Gender Inequality Index Forecast Nepal")
        forecast_gii(nepal_data_grouped)
    
    with col2:
        st.write("Digital Gender Gap")
        # Create a dropdown menu in the sidebar
        digital_gender_gap_option = st.selectbox(
            'Select digital gender gap',
            ('Internet Online','Internet Offline', 'Internet Both', 'Mobile Online', 'Mobile Offline', 'Mobile Both', 'Mobile GSMA')
        )
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

 ################################################################################################################################


    # Climate-Change Indicators
    st.write("Climate-Change Indicator Nepal")
    plot_climate_change_indicator(cci_df, climate_change_indicator_option, 'Value')
    st.write("Gender Gap Report Nepal")
    plot_ggr(ggr_data, selected_ggr_indicators)
    st.write("Gender Statistics Nepal")
    plot_gender_statistics(gender_statistics_data, selected_gender_statistics_indicator)


