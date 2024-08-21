from supabaseClient import client
from flask import Blueprint, render_template, request, redirect, url_for, jsonify, session

users = Blueprint('users', __name__)

supabase = client()

@users.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        try:
            response = supabase.auth.sign_in_with_password({
                "email": email, "password": password
            })

            if response.user:
                user_id = response.user.id
                session['user_id'] = user_id
                return redirect(url_for('dash.index'))

        except Exception as e:
            print(e)
            return render_template('login.html', error=e)
    return render_template('login.html')

@users.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        try:
            response = supabase.auth.sign_up({
                'email': email,
                'password': password
            })
            if response.user:
                return redirect(url_for('users.login'))
        except Exception as e:
            print(e)
            return render_template('register.html', error=e)
    return render_template('register.html')

@users.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('users.login'))