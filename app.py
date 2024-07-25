import os
import json
import mailbox
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from groq import Groq
from datetime import datetime
from dotenv import load_dotenv
from collections import defaultdict
from flask import Flask, render_template, request, jsonify, send_from_directory

load_dotenv()

app = Flask(__name__)
matplotlib.use('Agg')

# Configuration
UPLOAD_FOLDER = 'uploads/'
STATIC_FOLDER = 'static/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['STATIC_FOLDER'] = STATIC_FOLDER

# Initialize Groq client
client = Groq(
    api_key=os.getenv("GROQ_API_KEY"),
)

OWNER_EMAIL = 'user@example.com'
INVITATION_KEYWORDS = ['invite', 'invites', 'invited', 'inviting', 'invitation', 'introduce', 'introduction']

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
            {
                "role": "system",
                "content": "you are a sentiment analysis expert"
            },
            {
                "role": "user",
                "content": "Conduct sentiment analysis on the provided email content. The analysis must result in a single word indicating the sender's sentiment, chosen exclusively from \"positive,\" \"negative,\" or \"neutral.\" Please return only the selected word."
            },
            {
                "role": "user",
                "content": feed
            }
        ],
        temperature=1,
        max_tokens=1024,
        top_p=1,
        stream=False,
        stop=None,
    )
    
    sentiment_analysis = chat_completion.choices[0].message.content
    return sentiment_analysis

def groq_summary_relationship(feed):
    chat_completion = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {
                "role": "system",
                "content": "you are a sentiment analysis expert"
            },
            {
                "role": "user",
                "content": "Summarize my relationship with this contact based on provided email content in just one sentence. Don't start with Here is a summary of your relationship with this contact in one sentence:"
            },
            {
                "role": "user",
                "content": feed
            }
        ],
        temperature=1,
        max_tokens=1024,
        top_p=1,
        stream=False,
        stop=None,
    )

    summary = chat_completion.choices[0].message.content
    return summary

def visualize_data():
    save_location = app.config['STATIC_FOLDER']

    # Load the JSON data
    with open('email_data.json', 'r') as file:
        data = json.load(file)

    # Convert JSON data to DataFrame
    contacts = data['contacts']
    df = pd.DataFrame(contacts)

    # Ensure 'duration_known' is a numeric column
    df['duration_known'] = df['duration_known'].fillna(0).astype(float)

    # Set up the seaborn style
    sns.set(style="whitegrid")

    # Bar Chart of Number of Interactions
    plt.figure(figsize=(12, 6))
    sns.barplot(x='contact', y='number_of_interactions', data=df, palette='viridis')
    plt.xticks(rotation=45, ha='right')
    plt.title('Number of Interactions per Contact')
    plt.xlabel('Contact')
    plt.ylabel('Number of Interactions')
    plt.tight_layout()
    plt.savefig(save_location + 'number_of_interactions.png')  # Save the figure
    plt.close()

    # Histogram of Email Rate
    plt.figure(figsize=(8, 6))
    sns.histplot(df['email_rate'], bins=10, kde=True, color='blue')
    plt.title('Distribution of Email Rate')
    plt.xlabel('Email Rate')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(save_location + 'email_rate_distribution.png')  # Save the figure
    plt.close()

    # Pie Chart of Sentiment Analysis
    sentiment_counts = df['sentiment_analysis'].value_counts()
    plt.figure(figsize=(8, 8))
    plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', colors=sns.color_palette('pastel'))
    plt.title('Distribution of Sentiment Analysis')
    plt.savefig(save_location + 'sentiment_analysis_pie_chart.png')  # Save the figure
    plt.close()

    # Bar Chart of Number of Invitations
    plt.figure(figsize=(12, 6))
    sns.barplot(x='contact', y='number_of_invitations', data=df, palette='coolwarm')
    plt.xticks(rotation=45, ha='right')
    plt.title('Number of Invitations per Contact')
    plt.xlabel('Contact')
    plt.ylabel('Number of Invitations')
    plt.tight_layout()
    plt.savefig(save_location + 'number_of_invitations.png')  # Save the figure
    plt.close()

    # Scatter Plot of Email Rate vs. Number of Interactions
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='email_rate', y='number_of_interactions', data=df, hue='sentiment_analysis', palette='deep')
    plt.title('Email Rate vs. Number of Interactions')
    plt.xlabel('Email Rate')
    plt.ylabel('Number of Interactions')
    plt.legend(title='Sentiment Analysis')
    plt.tight_layout()
    plt.savefig(save_location + 'email_rate_vs_interactions.png')  # Save the figure
    plt.close()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_emails', methods=['POST'])
def process_emails():
    if 'mbox_file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['mbox_file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], 'dev.mbox')
        file.save(file_path)

        try:
            mbox = mailbox.mbox(file_path)

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

            with open('email_data.json', 'w') as json_file:
                json.dump(output_data, json_file, indent=4)

            # Call the visualization function
            visualize_data()

            return jsonify({'success': True})

        except Exception as e:
            return jsonify({'error': str(e)}), 500

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(app.config['STATIC_FOLDER'], filename)

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    if not os.path.exists(app.config['STATIC_FOLDER']):
        os.makedirs(app.config['STATIC_FOLDER'])
    app.run(debug=True)