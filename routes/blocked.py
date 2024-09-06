from supabase_config import client, retrieve_accounts
from flask import Blueprint, render_template, request, jsonify, session

blocked = Blueprint('blocked', __name__)

supabase = client()

@blocked.route('/', methods=['GET', 'POST'])
def addToBlocked():
    if request.method == 'POST':
        user_id = session['user_id']

        email_list = request.form.getlist('emails')
        bulk_email = []

        for email in email_list:
            obj = {
                "admin_id": user_id,
                "email_address": email
            }

            bulk_email.append(obj)

        try:
            response = (
                supabase.table("blocked_contacts")
                .upsert(
                    bulk_email,
                    on_conflict="email_address, admin_id"
                )
                .execute()
            )

            return jsonify({"success": "Emails successfully added to blocked list"}), 200
        
        except Exception as e:
            print(e)
            return render_template('blocked.html', error=str(e))
        
    accounts = retrieve_accounts()
    return render_template('blocked.html', accounts=accounts)