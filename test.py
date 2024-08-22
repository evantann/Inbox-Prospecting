from flask import Flask, render_template, redirect, url_for
from dash import Dash
import dash_core_components as dcc
import dash_html_components as html

# Initialize Flask app
app = Flask(__name__)

# Initialize Dash apps (one per account)
dashboards = {}

def create_dashboard(account_id):
    # Create a new Dash app instance
    dash_app = Dash(__name__, server=app, url_base_pathname=f'/dashboard/{account_id}/')
    
    # Define the layout of the dashboard
    dash_app.layout = html.Div([
        html.H1(f'Dashboard for Account {account_id}'),
        dcc.Graph(
            id='example-graph',
            figure={
                'data': [
                    {'x': [1, 2, 3], 'y': [1, 2, 3], 'type': 'line', 'name': f'Account {account_id}'},
                ],
                'layout': {
                    'title': f'Account {account_id} Data'
                }
            }
        )
    ])
    return dash_app

# Pre-register Dash apps for known account IDs
account_ids = [1, 2, 3]  # Example account IDs
for account_id in account_ids:
    dashboards[account_id] = create_dashboard(account_id)

@app.route('/')
def index():
    return redirect(url_for('render_dashboard', account_id=1))

@app.route('/dashboard/<int:account_id>/')
def render_dashboard(account_id):
    # Assuming the dashboard already exists
    if account_id in dashboards:
        return dashboards[account_id].index()
    else:
        return "Dashboard not found", 404

if __name__ == '__main__':
    app.run(debug=True)