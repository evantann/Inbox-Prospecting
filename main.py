import os
from flask import Flask
from routes.users import users
from routes.analyze import analyze
from routes.dashboard import dash
from flask import redirect, url_for

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)

app.register_blueprint(users, url_prefix='/users')
app.register_blueprint(analyze, url_prefix='/analyze')
app.register_blueprint(dash, url_prefix='/dash')

@app.route('/')
def root():
    return redirect(url_for('users.login'))

if __name__ == '__main__':
    app.run(debug=True)