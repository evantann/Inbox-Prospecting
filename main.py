import os
import pandas as pd
from dash import Dash, dcc, html, dash_table
from dash.dependencies import Input, Output
from flask import Flask, redirect, url_for, session, render_template
from routes.users import users
from routes.analyze import analyze
# from routes.dashboard import dashboard
from supabaseClient import client

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)

# Register blueprints
app.register_blueprint(users, url_prefix='/users')
app.register_blueprint(analyze, url_prefix='/analyze')
# app.register_blueprint(dashboard, url_prefix='/dashboard')  

supabase = client()

# Initialize Dash app
dash_app = Dash(__name__, server=app, url_base_pathname='/dash/')

# Updated Dash app layout
dash_app.layout = html.Div([
    dcc.Location(id='url', refresh=False),  # To capture the URL
    html.H1(id='dashboard-title'),
    
    # Section to display summary information
    html.Div(id='summary-container', children=[
        html.Div(className='summary-box', children=[
            html.H2('Number of Initiations'),
            html.P(id='num_initiations')
        ]),
        html.Div(className='summary-box', children=[
            html.H2('Total Number of Emails Exchanged'),
            html.P(id='num_emails')
        ]),
        html.Div(className='summary-box', children=[
            html.H2('Average Number of Interactions per Month'),
            html.P(id='avg_emails_per_month')
        ]),
        html.Div(className='summary-box', children=[
            html.H2('Average Response Time (hours)'),
            html.P(id='avg_response_time')
        ]),
        html.Div(className='summary-box', children=[
            html.H2('Average Sentiment Score'),
            html.P(id='avg_sentiment_score')
        ]),
        html.Div(className='summary-box', children=[
            html.H2('Average Personalization Score'),
            html.P(id='avg_personalization_score')
        ]),
    ]),

    # # Section to display plots
    # html.Div(id='plots-container', children=[
    #     html.Img(id='histogram_interaction', alt='Interaction Distribution'),
    #     html.Img(id='pie_chart_sentiment_analysis', alt='Pie Chart of Sentiment Analysis')
    # ]),

    # # Data table for detailed information
    # html.Div(id='dataframe', children=[
    #     html.H2('Table'),
    #     dash_table.DataTable(
    #         id='dataframe-table',
    #         columns=[
    #             {"name": "Field", "id": "field"},
    #             {"name": "Value", "id": "value"}
    #         ]
    #     )
    # ]),
])

# Dash app callback for updating the dashboard
@dash_app.callback(
    [Output('dashboard-title', 'children'),
     Output('num_initiations', 'children'),
     Output('num_emails', 'children'),
     Output('avg_emails_per_month', 'children'),
     Output('avg_response_time', 'children'),
     Output('avg_sentiment_score', 'children'),
     Output('avg_personalization_score', 'children'),
    #  Output('histogram_interaction', 'src'),
    #  Output('pie_chart_sentiment_analysis', 'src'),
    #  Output('dataframe-table', 'data')],
    ],
    [Input('url', 'pathname')]
)
def update_dashboard(pathname):
    account_id = pathname.split('/')[-2]

    data_query = (
        supabase.table("analysis")
        .select("*")
        .eq("account_id", account_id)
        .execute()
    )

    data = data_query.data
    df = pd.DataFrame(data)
    print(df)
    df_filtered = df.loc[
    (df['user_avg_response_time (hours)'] > 0) &
    (df['personalization_score'] > 0)
    ]

    num_contacts_user_initiated_true = int(df['user_initiated'].sum())
    average_response_time = round(df_filtered['user_avg_response_time (hours)'].mean(), 2)
    total_emails_exchanged = int(df['emails_exchanged'].sum())
    average_sentiment_score = round(df['sentiment_score'].mean(), 2)
    average_personalization_score = round(df_filtered['personalization_score'].mean(), 2)
    average_emails_per_month = round(df['interaction_frequency (emails per month)'].mean(), 2)

    title = f'Dashboard for Account {account_id}'

    return (title,
            num_contacts_user_initiated_true,
            total_emails_exchanged,
            average_emails_per_month,
            average_response_time,
            average_sentiment_score,
            average_personalization_score,
            # df['histogram_interaction_src'],
            # df['pie_chart_sentiment_analysis_src'],
            # df['dataframe_data']
            )

# Route to dashboard
@app.route('/dashboard/<account_id>')
def dashboard(account_id):

    dash_url = f"/dash/{account_id}/"
    user_id = session.get('user_id')
    print("This is the session", session)

    accounts_query = (
        supabase.table("accounts")
        .select("email", "id")
        .eq("admin_id", user_id)
        .execute()
    )

    accounts = accounts_query.data

    return render_template('dashboard.html', accounts=accounts, dash_url=dash_url)

# Main index route
@app.route('/dashboard')
def index():
    user_id = session.get('user_id')

    query = (
        supabase.table("accounts")
        .select("email", "id")
        .eq("admin_id", user_id)
        .execute()
    )

    accounts = query.data

    return render_template('dashboard.html', accounts=accounts)

# Root route to redirect to login
@app.route('/')
def root():
    return redirect(url_for('users.login'))

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)