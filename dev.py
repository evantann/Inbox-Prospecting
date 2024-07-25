from flask import Flask, jsonify, request, render_template
import os
import json
import mailbox
from groq import Groq
from datetime import datetime
from dotenv import load_dotenv
from collections import defaultdict

load_dotenv()

OWNER_EMAIL = 'user@example.com'
INVITATION_KEYWORDS = ['invite', 'invites', 'invited', 'inviting', 'invitation', 'introduce', 'introduction']

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

app = Flask(__name__)

def parse_date(date_str):
    try:
        return datetime.strptime(date_str, '%a, %d %b %Y %H:%M:%S %z')
    except ValueError:
        return None

def is_invitation(subject, body):
    subject = subject.lower() if subject else ''
    body = body.lower() if body else ''
    return any(keyword in subject or keyword in body for keyword in INVITATION_KEYWORDS)

def calculate_response_time(email_groups):
    threads = defaultdict(list)
    response_times = []

    for contact, emails in email_groups.items():
        for email in emails:
            subject = email['Subject']
            if subject and ('Re:' in subject or 'RE' in subject):
                threads[subject].append(email)

    for thread_emails in threads.values():
        sorted_emails = sorted(thread_emails, key=lambda x: parse_date(x['Date']) or datetime.min)
        for i in range(1, len(sorted_emails)):
            current_email = sorted_emails[i]
            previous_email = sorted_emails[i - 1]
            if OWNER_EMAIL in current_email['From'] and OWNER_EMAIL not in previous_email['From']:
                response_time = (parse_date(current_email['Date']) - parse_date(previous_email['Date'])).seconds
                response_time /= 3600
                response_times.append(response_time)

    if response_times:
        average_response_time = sum(response_times) / len(response_times)
        return average_response_time
    else:
        return None

def process_message(message, email_groups, interaction_counts, invitation_counts, first_email_dates, last_email_dates):
    from_address = message['From']
    to_addresses = message.get_all('To', [])
    date_str = message['Date']
    date = parse_date(date_str)
    subject = message['Subject']
    body = message.get_payload()

    contacts = []

    if from_address and OWNER_EMAIL not in from_address:
        contacts.append(from_address.split('<')[0].strip())

    for to_address in to_addresses:
        for address in to_address.split(','):
            address = address.strip()
            if OWNER_EMAIL not in address:
                contacts.append(address.split('<')[0].strip())

    for contact in contacts:
        email_groups[contact].append({
            'From': from_address,
            'To': to_addresses,
            'Subject': subject,
            'Date': date_str,
            'Body': body
        })
        interaction_counts[contact] += 1

        if contact == from_address and is_invitation(subject, body):
            invitation_counts[contact] += 1

        if date:
            if first_email_dates[contact] is None or date < first_email_dates[contact]:
                first_email_dates[contact] = date
            if last_email_dates[contact] is None or date > last_email_dates[contact]:
                last_email_dates[contact] = date

def prepare_output(email_groups, interaction_counts, invitation_counts, first_email_dates, last_email_dates, sentiment_analyses, summary_relationships):
    return {
        'number_of_unique_contacts': len(email_groups),
        'average_response_time': calculate_response_time(email_groups),
        'contacts': [
            {
                'contact': contact,
                'number_of_interactions': interaction_counts[contact],
                'duration_known': (last_email_dates[contact] - first_email_dates[contact]).days if first_email_dates[contact] and last_email_dates[contact] else None,
                'number_of_invitations': invitation_counts[contact],
                'email_rate': (interaction_counts[contact] / ((last_email_dates[contact] - first_email_dates[contact]).days + 1)) if first_email_dates[contact] and last_email_dates[contact] else None,
                'is_influencer': invitation_counts[contact] > 1,
                'emails': email_groups[contact],
                'sentiment_analysis': sentiment_analyses[contact],
                'summary_relationship': summary_relationships[contact]
            }
            for contact in email_groups
        ]
    }

def groq_sentiment_analysis(feed):
    chat_completion = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "system", "content": "you are a sentiment analysis expert"},
            {"role": "user", "content": "Conduct sentiment analysis on the provided email content. The analysis must result in a single word indicating the sender's sentiment, chosen exclusively from \"positive,\" \"negative,\" or \"neutral.\" Please return only the selected word."},
            {"role": "user", "content": feed}
        ],
        temperature=1,
        max_tokens=1024,
        top_p=1
    )
    sentiment_analysis = chat_completion.choices[0].message.content
    return sentiment_analysis

def groq_summary_relationship(feed):
    chat_completion = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "system", "content": "you are a sentiment analysis expert"},
            {"role": "user", "content": "Summarize my relationship with this contact based on provided email content in just one sentence. Don't start with Here is a summary of your relationship with this contact in one sentence:"},
            {"role": "user", "content": feed}
        ],
        temperature=1,
        max_tokens=1024,
        top_p=1
    )
    summary = chat_completion.choices[0].message.content
    return summary

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_emails', methods=['POST'])
def process_emails():
    try:
        # Assuming you are sending the mbox file as part of the POST request
        mbox_file = request.files.get('mbox_file')
        if not mbox_file:
            return jsonify({'error': 'No mbox file provided'}), 400

        mbox = mailbox.mbox(mbox_file)
        email_groups = defaultdict(list)
        interaction_counts = defaultdict(int)
        invitation_counts = defaultdict(int)
        first_email_dates = defaultdict(lambda: None)
        last_email_dates = defaultdict(lambda: None)
        sentiment_analyses = defaultdict(str)
        summary_relationships = defaultdict(str)

        for message in mbox:
            process_message(message, email_groups, interaction_counts, invitation_counts, first_email_dates, last_email_dates)

        for contact, emails in email_groups.items():
            feed = " ".join([email['Body'] for email in emails])
            sentiment_analysis = groq_sentiment_analysis(feed)
            sentiment_analyses[contact] = sentiment_analysis
            summary_relationship = groq_summary_relationship(feed)
            summary_relationships[contact] = summary_relationship

        output_data = prepare_output(email_groups, interaction_counts, invitation_counts, first_email_dates, last_email_dates, sentiment_analyses, summary_relationships)

        return jsonify(output_data)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)