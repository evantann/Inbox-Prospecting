import os
from flask import session
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")

def client():
    return create_client(supabase_url, supabase_key)

supabase = client()

def retrieve_accounts():
    user_id = session.get('user_id')

    query = (
        supabase.table("accounts")
        .select("email", "id")
        .eq("admin_id", user_id)
        .execute()
    )
    
    accounts = query.data
    return accounts