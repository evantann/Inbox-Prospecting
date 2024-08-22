
import os
import dash_core_components as dcc
import dash_html_components as html
from dash import Dash
from flask import Flask
from routes.users import users
from routes.analyze import analyze
from routes.dashboard import dashboard
from flask import redirect, url_for

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)

app.register_blueprint(users, url_prefix='/users')
app.register_blueprint(analyze, url_prefix='/analyze')
app.register_blueprint(dashboard, url_prefix='/dashboard')  

def create_dashboard(account_id):
    # Create a new Dash app instance
    dash_app = Dash(__name__, server=app, url_base_pathname=f'/dash/{account_id}/')
    
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

@app.route('/')
def root():
    return redirect(url_for('users.login'))

if __name__ == '__main__':
    app.run(debug=True)