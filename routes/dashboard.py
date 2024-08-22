from supabaseClient import client
from flask import Blueprint, render_template, session

dashboard = Blueprint('dashboard', __name__)

supabase = client()

@dashboard.route('/')
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

@dashboard.route('/<account_id>') 
def render_dashboard(account_id):

    user_id = session.get('user_id')

    query = (
        supabase.table("accounts")
        .select("email", "id")
        .eq("admin_id", user_id)
        .execute()
    )

    data = (
        supabase.table("analysis")
        .select("*")
        .eq("account_id", account_id)
        .execute()
    )

    accounts = query.data
    data = data.data
    dash_url = f"/dash/{account_id}/"

    return render_template('dashboard.html', accounts=accounts, data=data, dash_url=dash_url)