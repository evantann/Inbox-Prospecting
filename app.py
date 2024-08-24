import os
import redis
import pandas as pd
import plotly.express as px
import dash_bootstrap_components as dbc
from routes.users import users
from datetime import timedelta
from flask_session import Session
from routes.analyze import analyze
from dash.dependencies import Input, Output
from dash import Dash, dcc, html, dash_table
from supabase_config import client, retrieve_accounts
from flask import Flask, redirect, url_for, render_template

app = Flask(__name__)

app.config['SECRET_KEY'] = 'your_secret_key'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=1) # SESSION_PERMANENT must be set to true for this config to apply
app.config['SESSION_PERMANENT'] = True
app.config['SESSION_TYPE'] = 'redis'
app.config['SESSION_REDIS'] = redis.StrictRedis(host='localhost', port=6379)
app.config['SESSION_USE_SIGNER'] = True

Session(app)

app.register_blueprint(users, url_prefix='/users')
app.register_blueprint(analyze, url_prefix='/analyze')

if not os.path.exists('uploads'):
    os.makedirs('uploads')

supabase = client()

dash_app = Dash(__name__, server=app, url_base_pathname='/dash/', external_stylesheets=[dbc.themes.BOOTSTRAP])

# Dash styling
custom_css = {
    "card": {
        "borderRadius": "10px",
        "boxShadow": "0 4px 8px 0 rgba(0, 0, 0, 0.2)",
    },
    "card-title": {
        "fontSize": "1.25rem",
        "fontWeight": "bold",
    },
    "card-text": {
        "fontSize": "1rem",
        "color": "#495057",
    },
    "container": {
        "backgroundColor": "#f8f9fa",
        "padding": "20px",
        "borderRadius": "15px",
    },
    "title": {
        "color": "#343a40",
    },
}

dash_app.layout = dbc.Container([
    dcc.Location(id='url', refresh=False),
    dbc.Row([
        dbc.Col(html.H1(id='dashboard-title', className='text-center my-4', style=custom_css["title"]), width=12)
    ]),
    
    # Section to display summary information (fit on one line)
    dbc.Row(id='summary-container', children=[
        dbc.Col(className='summary-box', children=[
            dbc.Card([
                dbc.CardBody([
                    html.H5('Number of Initiations', className='card-title', style=custom_css["card-title"]),
                    html.P(id='num_initiations', className='card-text', style=custom_css["card-text"])
                ])
            ], className='shadow-sm mb-4', style=custom_css["card"]),
        ], width=2),
        dbc.Col(className='summary-box', children=[
            dbc.Card([
                dbc.CardBody([
                    html.H5('Total Number of Emails Exchanged', className='card-title', style=custom_css["card-title"]),
                    html.P(id='num_emails', className='card-text', style=custom_css["card-text"])
                ])
            ], className='shadow-sm mb-4', style=custom_css["card"]),
        ], width=2),
        dbc.Col(className='summary-box', children=[
            dbc.Card([
                dbc.CardBody([
                    html.H5('Average Interactions per Month', className='card-title', style=custom_css["card-title"]),
                    html.P(id='avg_emails_per_month', className='card-text', style=custom_css["card-text"])
                ])
            ], className='shadow-sm mb-4', style=custom_css["card"]),
        ], width=2),
        dbc.Col(className='summary-box', children=[
            dbc.Card([
                dbc.CardBody([
                    html.H5('Average Response Time (hours)', className='card-title', style=custom_css["card-title"]),
                    html.P(id='avg_response_time', className='card-text', style=custom_css["card-text"])
                ])
            ], className='shadow-sm mb-4', style=custom_css["card"]),
        ], width=2),
        dbc.Col(className='summary-box', children=[
            dbc.Card([
                dbc.CardBody([
                    html.H5('Average Sentiment Score', className='card-title', style=custom_css["card-title"]),
                    html.P(id='avg_sentiment_score', className='card-text', style=custom_css["card-text"])
                ])
            ], className='shadow-sm mb-4', style=custom_css["card"]),
        ], width=2),
        dbc.Col(className='summary-box', children=[
            dbc.Card([
                dbc.CardBody([
                    html.H5('Average Personalization Score', className='card-title', style=custom_css["card-title"]),
                    html.P(id='avg_personalization_score', className='card-text', style=custom_css["card-text"])
                ])
            ], className='shadow-sm mb-4', style=custom_css["card"]),
        ], width=2),
    ], style=custom_css["container"]),
    
    # Row for the histogram, pie chart, and data table
    dbc.Row([
        dbc.Col(dcc.Graph(id='emails_histogram'), width=6),
        dbc.Col(dcc.Graph(id='sentiment_pie_chart'), width=6),
    ], style=custom_css["container"]),

    # Data Table with index column, search, and sort functionality
    dbc.Row([
        dbc.Col(dash_table.DataTable(
            id='data_table',
            columns=[],
            data=[],
            filter_action='native',
            sort_action='native',
            sort_mode='multi',
            style_table={'overflowX': 'auto'},
            style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
            style_cell={'textAlign': 'left', 'padding': '5px', 'whiteSpace': 'normal', 'height': 'auto'}
        ), width=12),
    ], style=custom_css["container"]),

], fluid=True)

# Dash app callback for updating the dashboard
@dash_app.callback([
    Output('dashboard-title', 'children'),
    Output('num_initiations', 'children'),
    Output('num_emails', 'children'),
    Output('avg_emails_per_month', 'children'),
    Output('avg_response_time', 'children'),
    Output('avg_sentiment_score', 'children'),
    Output('avg_personalization_score', 'children'),
    Output('emails_histogram', 'figure'),
    Output('sentiment_pie_chart', 'figure'),
    Output('data_table', 'columns'),
    Output('data_table', 'data')],
    [Input('url', 'pathname')]
)
def update_dashboard(pathname):
    account_id = pathname.split('/')[-2]

    data_query = supabase.table("analysis").select("*").eq("account_id", account_id).execute()
    data = data_query.data

    df = pd.DataFrame(data)

    df['Index'] = df.index + 1
    df = df[['Index'] + [col for col in df.columns if col != 'Index']]

    df_filtered = df.drop(columns=['account_id', 'id'])

    table_columns = [{"name": i, "id": i} for i in df_filtered.columns]
    table_data = df_filtered.to_dict('records')

    df_filtered_stats = df.loc[
        (df['user_avg_response_time_hours'] > 0) &
        (df['personalization_score'] > 0)
    ]

    df_top_100 = df.sort_values(by='emails_exchanged', ascending=False).head(100)
    fig_histogram = px.histogram(df_top_100, x='emails_exchanged', nbins=20, title='Distribution of Emails Exchanged')

    sentiment_counts = df['relationship_summary'].value_counts()
    fig_pie_chart = px.pie(
        sentiment_counts, 
        values=sentiment_counts.values, 
        names=sentiment_counts.index, 
        title='Sentiment Analysis Distribution'
    )

    response = (
        supabase.table("accounts")
        .select("email")
        .eq("id", account_id)
        .execute()
    )

    email = response.data[0]['email']

    title = f'Inbox for {email}'

    num_contacts_user_initiated_true = int(df['user_initiated'].sum())
    average_response_time = round(df_filtered_stats['user_avg_response_time_hours'].mean(), 2)
    total_emails_exchanged = int(df['emails_exchanged'].sum())
    average_sentiment_score = round(df['sentiment_score'].mean(), 2)
    average_personalization_score = round(df_filtered_stats['personalization_score'].mean(), 2)
    average_emails_per_month = round(df['emails_per_month'].mean(), 2)

    return (title,
            num_contacts_user_initiated_true,
            total_emails_exchanged,
            average_emails_per_month,
            average_response_time,
            average_sentiment_score,
            average_personalization_score,
            fig_histogram,
            fig_pie_chart,
            table_columns,
            table_data)

# Route to dashboard
@app.route('/dashboard/<account_id>')
def dashboard(account_id):
    dash_url = f"/dash/{account_id}/"
    accounts = retrieve_accounts()
    return render_template('dashboard.html', accounts=accounts, dash_url=dash_url)

# Main index route
@app.route('/dashboard')
def index():
    accounts = retrieve_accounts()
    return render_template('dashboard.html', accounts=accounts)

# Root route to redirect to login
@app.route('/')
def root():
    return redirect(url_for('users.login'))

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)