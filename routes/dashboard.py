from supabaseClient import client
from flask import Blueprint, render_template, request, redirect, url_for, jsonify, session

dash = Blueprint('dash', __name__)

supabase = client()

@dash.route('/')
def index():
    user_id = session.get('user_id')

    query = (
        supabase.table("accounts")
        .select("email", "admin_id")
        .eq("admin_id", user_id)
        .execute()
    )

    accounts = query.data

    return render_template('dash.html', accounts=accounts)

@dash.route('/<email>')
def dashboard(email):
    user_id = session.get('user_id')

    query = (
        supabase.table("accounts")
        .select("email", "admin_id")
        .eq("admin_id", user_id)
        .execute()
    )

    accounts = query.data

    return render_template('dash.html', email=email, accounts=accounts)