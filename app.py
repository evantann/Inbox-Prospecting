import os
import re
import json
import spacy
import torch
import mailbox
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datetime import datetime
from textblob import TextBlob
from collections import defaultdict, Counter
from flask import Flask, render_template, request, jsonify
from transformers import AutoModelForSequenceClassification, AutoTokenizer

app = Flask(__name__)
matplotlib.use('Agg') # Required for matplotlib to work with Flask

UPLOAD_FOLDER = 'uploads/'
STATIC_FOLDER = 'static/'

BLOCKED_CONTACTS = ['reply', 'support', 'notification', 'human resources', 'rewards', 'orders', 'alerts', 'talent', 'recruit', 'info', 'email', 'customer', 'account', 'admission', 'store', 'club', 'subscription', 'news', 'newsletter', 'product', 'updates', 'help', 'assistance', 'jury', 'careers', 'sale', 'response', 'guest', 'user', 'robot', 'confirm', 'automate', 'website', 'notice']

INVITATION_KEYWORDS = {
    'invite', 'invites', 'invited', 'inviting', 'invitation', 'introduce', 'introduction', 'RSVP', 'like to meet', 'attend', 'event', 'participate'
}
STOP_WORDS = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves',
    'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
    'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an',
    'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about',
    'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up',
    'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when',
    'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
    'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should',
    'now', 'hello', 'dear', 'hi', 'hey', 'regards', 'thanks', 'thank', 'best', 'kind', 'warm', 'sincerely', 'yours', 'tomorrow',
    'cheers', 'everything', 'next', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday', 'need', 'please',
    'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december',
    '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', 'am', 'pm', 'it', 'its', 'it\'s', 'they', 'them', 'their', 'theirs', 'yes',
    'ipad', 'iphone', 'would', 'just', 'i\'m', 'i\'d' , 'i\'ve', 'you\'re', 'i\'ll'
}

def parse_date(date_string):
    date_string = re.sub(r'\s\([A-Z]{2,}\)$', '', date_string)

    formats = [
        '%a, %d %b %Y %H:%M:%S %z',
        '%a %d %b %Y %H:%M:%S %z',
        '%d %b %Y %H:%M:%S %z'
    ]
    
    for format_string in formats:
        try:
            return datetime.strptime(date_string, format_string)
        except ValueError:
            continue

    print(f"Could not parse date: {date_string}")
    
    return None

def is_invitation(subject, body):
    try:
        subject = subject.lower()
        body = body.lower()
        return any(keyword in subject or keyword in body for keyword in INVITATION_KEYWORDS)
    except Exception as e:
        print(f"Error in is_invitation: {e}")

def sentiment_analysis(sentiment_scores, relationship_summaries, address, emails):
    sentiment_scores_list = []

    try:
        for email in emails:
            blob = TextBlob(email['Body'])
            polarity = blob.sentiment.polarity
            sentiment_scores_list.append(polarity)
        if sentiment_scores_list:
            sentiment_score = sum(sentiment_scores_list) / len(sentiment_scores_list)
            sentiment_summary = 'positive' if sentiment_score > 0 else 'negative' if sentiment_score < 0 else 'neutral'
            sentiment_scores[address] = (sentiment_score)
            relationship_summaries[address] = sentiment_summary

    except Exception as e:
        print(f"Error in sentiment_analysis: {e}")

def organize_by_thread(threads, address, emails):
    try:
        for email in emails:
            subject = email['Subject']
            if 'Re: ' in subject or 'RE: ' in subject:
                threads[address][subject].append(email)
    except Exception as e:
        print(f"Error in organize_by_thread: {e}")

def calculate_user_response_times(threads, user_response_times_by_contact, median_response_times, user_email):
    try:
        for contact, threadz in threads.items():
            for emails in threadz.values():
                for i in range(1, len(emails)):
                    current_email = emails[i]
                    previous_email = emails[i - 1]
                    
                    if user_email in current_email['From'] and user_email not in previous_email['From']:
                        current_date = parse_date(current_email['Date'])
                        previous_date = parse_date(previous_email['Date'])
                        if current_date and previous_date:
                            response_time = (current_date - previous_date).seconds
                            response_time /= 3600
                            user_response_times_by_contact[contact].append(response_time)

        for contact, times in user_response_times_by_contact.items():
            if times:
                median_response_times[contact] = np.median(times)
                user_response_times_by_contact[contact] = sum(times) / len(times)

    except Exception as e:
        print(f"Error in calculating response times: {e}")

def calculate_contact_response_times(threads, contact_response_times, user_email):
    try:
        for contact, threadz in threads.items():
            for emails in threadz.values():
                for i in range(1, len(emails)):
                    current_email = emails[i]
                    previous_email = emails[i - 1]
                    
                    if user_email in previous_email['From'] and user_email not in current_email['From']:
                        current_date = parse_date(current_email['Date'])
                        previous_date = parse_date(previous_email['Date'])
                        if current_date and previous_date:
                            response_time = (current_date - previous_date).seconds
                            response_time /= 3600
                            contact_response_times[contact].append(response_time)

        for contact, times in contact_response_times.items():
            if times:
                contact_response_times[contact] = sum(times) / len(times)

    except Exception as e:
        print(f"Error in calculating response times: {e}")

def calculate_follow_up_rate(follow_up, address, emails, user_email):
    try:
        emails_from_contact_count = sum(1 for email in emails if address in email['From'])
        follow_up_count = 0

        for i in range(1, len(emails)):
            current_email = emails[i]
            previous_email = emails[i - 1]

            if user_email in current_email['From'] and user_email not in previous_email['From']:
                follow_up_count += 1

        follow_up[address] = (follow_up_count / emails_from_contact_count * 100) if emails_from_contact_count > 0 else 0
        
    except Exception as e:
        print(f"Error in calculating follow up rate: {e}")

def calculate_interaction_frequency(monthly_interactions, interactions_each_month, address, emails):
    try:
        for email in emails:
            date_str = email["Date"]
            try:
                date = parse_date(date_str)
                if date:
                    month = date.strftime('%m')
                    month = int(month.lstrip('0'))
                    interactions_each_month[address][month] += 1
            except ValueError as e:
                print(f"Error parsing date in interaction_frequency: {e}")

        for email_address, months in interactions_each_month.items():
            total_sum = 0
            non_zero_months = 0
            for month in months:
                total_sum += month
                if month > 0:
                    non_zero_months += 1
            monthly_interactions[email_address] = total_sum / non_zero_months if non_zero_months > 0 else 0

    except Exception as e:
        print(f"Error in calculating interaction frequency: {e}")

def user_initiated(user_initiation, address, emails, user_email):
    if not emails:
        return
    try:
        user_initiation[address] = user_email in emails[0]['From']
    except Exception as e:
        print(f"Error in determining user initiation: {e}")

def calculate_personalization_score(nlp, personalization_scores, address, emails, user_email):
    try:
        for email in emails:
            if user_email in email['From']:
                doc = nlp(email['Body'])
                for ent in doc.ents:
                    if ent.label_ == 'PERSON':
                        personalization_scores[address] += 1
    except Exception as e:
        print(f"Error in calculating personalization score: {e}")

def find_keywords(keywords, address, emails):
    try:
        all_messages = " ".join([email['Body'] for email in emails])
        all_messages = all_messages.lower()

        punctuation_marks = ['.', ',', '!', '?', ':', ';', '\n', '\t', '\r', '(', ')', '[', ']', '{', '}', '<', '>', '"', '—', '-', '_', '/', '\\', '|', '@', '#', '$', '%', '^', '&', '*', '+', '=', '~', '`']
        for marks in punctuation_marks:
            all_messages = all_messages.replace(marks, '')

        all_messages = all_messages.split()
        filtered_messages = [word for word in all_messages if word not in STOP_WORDS]
        word_counts = Counter(filtered_messages)
        most_common_words = word_counts.most_common(5)
        top_words = ', '.join([word for word, count in most_common_words])
        keywords[address] = top_words
        
    except Exception as e:
        print(f"Error in finding keywords: {e}")

def process_message(message, email_meta, email_content, contact_names, interaction_counts, invitation_counts, first_email_dates, last_email_dates, user_email):
    try:
        contacts = []

        if message['From']:
            from_address = message['From']
        else:
            from_address = ''

        if message['To']:
            to_addresses = message['To']
        else:
            to_addresses = ''

        if message['Date']:
            date_str = message['Date']
            date = parse_date(date_str)
        else:
            date_str = ''
            date = None

        if message['Subject']:
            subject = message['Subject']
        else:
            subject = ''

        if message['Body']:
            body = message['Body']
        else:
            body = ''

        if user_email not in from_address: # if the owner is not the sender, add the sender to the contact list
            split_address = from_address.split('<')
            if len(split_address) > 1: #  Two name-address formats: First Last <email> OR email
                name = split_address[0].strip()
                address = split_address[1].replace('>', '').strip()
                contacts.append(address)
                if address not in contact_names:
                    contact_names[address] = name
        elif user_email in from_address:
            split_to_addresses = to_addresses.split(',')
            if len(split_to_addresses) > 1: # multiple recipients
                for address in split_to_addresses:
                    if user_email not in address:
                        address = address.strip()
                        contacts.append(address)
            else:
                split_address = to_addresses.split('<')
                if len(split_address) > 1:
                    name = split_address[0].strip()
                    address = split_address[1].replace('>', '').strip()
                    if user_email not in address: # handle case where owner is the sender and recipient
                        contacts.append(address)
                        if address not in contact_names:
                            contact_names[address] = name

        for address in contacts:
            email_meta[address].append({
                'From': from_address,
                'To': to_addresses,
                'Subject': subject,
                'Date': date_str,
                'Body': body
            })

            body = body.replace('\r\n', ' ')
            body = body.replace('\n', ' ')
            email_content[address].append(body)

            interaction_counts[address] += 1

            if address in from_address and is_invitation(subject, body):
                invitation_counts[address] += 1

            if date:
                if first_email_dates[address] is None or date < first_email_dates[address]:
                    first_email_dates[address] = date
                if last_email_dates[address] is None or date > last_email_dates[address]:
                    last_email_dates[address] = date

    except Exception as e:
        print(f"Error processing message: {e}")

def generate_tabular_data(email_meta, email_content, contact_names, interaction_counts, invitation_counts, first_email_dates, last_email_dates, sentiment_scores, relationship_summaries, monthly_interactions, user_initiation, personalization_scores, follow_up_rates, keywords, contact_response_times, user_response_times_by_contact, median_response_times, user_email):
    try:
        data = [{
            'contact': contact_names.get(contact, 'N/A'),
            'email_address': contact if contact else 'N/A',
            'emails_exchanged': interaction_counts.get(contact, 0),
            'number_of_invitations_received': invitation_counts.get(contact, 0),
            'duration_known (months)': ((last_email_dates.get(contact) - first_email_dates.get(contact)).days) / 30 if first_email_dates.get(contact) and last_email_dates.get(contact) else 0,
            'sentiment_score': sentiment_scores.get(contact, 0),
            'relationship_summary': relationship_summaries.get(contact, 'N/A'),
            'user_initiated': user_initiation.get(contact, False    ),
            'interaction_frequency (emails per month)': monthly_interactions.get(contact, 0),
            'follow_up_rate (%)': follow_up_rates.get(contact, 0),
            "contact_response_time (hours)": contact_response_times.get(contact, 0),
            'average_response_time (hours)': user_response_times_by_contact.get(contact, 0),
            'median_response_time (hours)': median_response_times.get(contact, 0),
            'keywords': keywords.get(contact, 'N/A'),
            'personalization_score': personalization_scores.get(contact, 0) / (sum(1 for email in email_meta.get(contact, []) if user_email in email['From']) if sum(1 for email in email_meta.get(contact, []) if user_email in email['From']) > 0 else 1)
        } for contact in email_content]
        return data
    except Exception as e:
        print(f"Error generating tabular data: {e}")
        return []

def main(data, user_email, nlp):
    try:
        keywords = defaultdict(list)
        email_meta = defaultdict(list)
        email_content = defaultdict(list)
        contact_names = defaultdict(str)
        sentiment_scores = defaultdict(str)   
        relationship_summaries = defaultdict(str)
        personalization_scores = defaultdict(int)
        user_response_times_by_contact = defaultdict(list)
        contact_response_times = defaultdict(list)
        median_response_times = defaultdict(int)
        monthly_interactions = defaultdict(int)
        interaction_counts = defaultdict(int)
        invitation_counts = defaultdict(int)
        follow_up_rates = defaultdict(int)
        user_initiation = defaultdict(bool)
        last_email_dates = defaultdict(lambda: None)
        first_email_dates = defaultdict(lambda: None)
        threads = defaultdict(lambda: defaultdict(list))
        interactions_each_month = defaultdict(lambda: [0] * 13)
        count = 0
        
        for entry in data: 
            process_message(entry, email_meta, email_content, contact_names, interaction_counts, invitation_counts, first_email_dates, last_email_dates, user_email)

        for address, emails in email_meta.items():
            count += 1
            print(f'Processing contact {count} of {len(email_meta)}')
            sentiment_analysis(sentiment_scores, relationship_summaries, address, emails)
            calculate_interaction_frequency(monthly_interactions, interactions_each_month, address, emails)
            calculate_personalization_score(nlp, personalization_scores, address, emails, user_email)
            find_keywords(keywords, address, emails)

            valid_emails = [email for email in emails if parse_date(email['Date'])]
            sorted_emails = sorted(valid_emails, key=lambda x: parse_date(x['Date']))
            organize_by_thread(threads, address, sorted_emails)
            calculate_follow_up_rate(follow_up_rates, address, sorted_emails, user_email)
            user_initiated(user_initiation, address, sorted_emails, user_email)
        
        calculate_user_response_times(threads, user_response_times_by_contact, median_response_times, user_email)
        calculate_contact_response_times(threads, contact_response_times, user_email)
        
        output_data = generate_tabular_data(
            email_meta, email_content, contact_names, interaction_counts, invitation_counts,
            first_email_dates, last_email_dates, sentiment_scores, relationship_summaries, monthly_interactions, user_initiation,
            personalization_scores, follow_up_rates, keywords, contact_response_times, user_response_times_by_contact, median_response_times, user_email
        )

        print(f'Data has been written to {user_email}.json')

        return output_data

    except Exception as e:
        print(f'An error occurred in main(): {e}')

def clean_headers(msg):
    try:
        allowed_headers = ['To', 'From', 'Subject', 'Date']
        cleaned_msg = {key: str(msg.get(key)) for key in allowed_headers if msg.get(key)}
        return cleaned_msg
    except Exception as e:
        print(f'Error cleaning headers: {e}')

def predict_spam(text, model, tokenizer):
    inputs = tokenizer(text, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probabilities = F.softmax(logits, dim=-1)
    predicted_class = torch.argmax(probabilities, dim=-1).item()
    labels = ['not spam', 'spam']
    predicted_label = labels[predicted_class]
    
    return predicted_label

def clean_payload(payload, forwarded_message_pattern, on_wrote_pattern, from_pattern):
    try:
        if re.match(forwarded_message_pattern, payload, re.DOTALL):
            payload = payload
        elif re.match(on_wrote_pattern, payload, re.DOTALL):
            match = re.match(on_wrote_pattern, payload, re.DOTALL)
            if match:
                payload = match.group(1)
        
        elif re.match(from_pattern, payload, re.DOTALL):
            match = re.match(from_pattern, payload, re.DOTALL)
            if match:
                payload = match.group(1)
        
        return payload

    except Exception as e:
        print(f'Error cleaning payload: {e}')

def extract_mbox(input_mbox, model, tokenizer):

    on_wrote_pattern = r'^(.*?)(On .*? wrote:)'
    from_pattern = r'^(.*?)(?=From:.*<[^>]+>)'
    forwarded_message_pattern = r'^-{10,} Forwarded message -{10,}$'

    try:
        processed_messages = []
        in_mbox = mailbox.mbox(input_mbox)
        count = 0
        
        for message in in_mbox:
            print(f"Extracting message #{count} of {len(in_mbox)}")
            count += 1

            from_address = str(message.get('From'))
            if any(word in from_address.lower() for word in BLOCKED_CONTACTS):
                continue

            to_address = str(message.get('To'))
            if any(word in to_address.lower() for word in BLOCKED_CONTACTS):
                continue
            
            if message.is_multipart():
                for part in message.walk():
                    if part.get_content_type() != 'text/plain':
                        continue

                    payload = str(part.get_payload(decode=True).decode('utf-8', errors='replace'))
                    if not payload.strip():
                        continue
                
                    if 'unsubscribe' in payload.lower():
                        continue
                    
                    payload = clean_payload(payload, forwarded_message_pattern, on_wrote_pattern, from_pattern)
            
                    try:
                        result = predict_spam(payload, model, tokenizer)
                        if result == 'spam':
                            continue
                    except Exception as e:
                        continue
            
                    email_dict = {
                        'From': str(message.get('From')),
                        'To': str(message.get('To')),
                        'Subject': str(message.get('Subject')),
                        'Date': str(message.get('Date')),
                        'Body': payload
                    }
                    
                    processed_messages.append(email_dict)
                    break

            else:
                if message.get_content_type() != 'text/plain':
                    continue

                email_dict = clean_headers(message)
                payload = str(message.get_payload(decode=True).decode('utf-8', errors='replace'))
                if not payload.strip():
                    continue

                if 'unsubscribe' in payload.lower():
                    continue
                
                payload = clean_payload(payload, forwarded_message_pattern, on_wrote_pattern, from_pattern)

                try:
                    result = predict_spam(payload, model, tokenizer)
                    if result == 'spam':
                        continue
                except Exception as e:
                    continue
                
                email_dict['Body'] = payload
                processed_messages.append(email_dict)


        return processed_messages
    
    except Exception as e:
        print(f'Error extracting mbox: {e}')


def generate_plots(data):
    try:
        df = pd.DataFrame(data)

        df_filtered = df.loc[
            (df['average_response_time (hours)'] > 0) &
            (df['personalization_score'] > 0)
        ]

        df_top_100 = df.sort_values(by='emails_exchanged', ascending=False).head(100)

        # Histogram of Average Response Times
        # plt.figure(figsize=(12, 6))
        # sns.histplot(df_filtered['average_response_time (hours)'], bins=10, kde=True, color='blue')
        # plt.title('Histogram of Average Response Times')
        # plt.xlabel('Average Response Time (hours)')
        # plt.ylabel('Frequency')
        # plt.grid(True)
        # plt.savefig('static/histogram_response_times.png')
        # plt.close()

        # Histogram of Personalization Scores
        # plt.figure(figsize=(12, 6))
        # sns.histplot(df_filtered['personalization_score'], bins=10, kde=True, color='green')
        # plt.title('Histogram of Personalization Scores')
        # plt.xlabel('Personalization Score')
        # plt.ylabel('Frequency')
        # plt.grid(True)
        # plt.savefig('static/histogram_personalization_scores.png')
        # plt.close()

        # Histogram of Number of Interactions
        plt.figure(figsize=(12, 6))
        sns.histplot(df_top_100['emails_exchanged'], bins=50, kde=True, color='purple')
        plt.title('Histogram of Number of Interactions')
        plt.xlabel('Number of Interactions')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.savefig('static/histogram_interactions.png')
        plt.close()

        # Pie Chart of Relationship Summary
        sentiment_counts = df['relationship_summary'].value_counts()
        plt.figure(figsize=(8, 8))
        plt.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', colors=['#66c2a5', '#fc8d62', '#8da0cb'])
        plt.title('Sentiment Analysis Distribution')
        plt.savefig('static/pie_chart_sentiment_analysis.png')
        plt.close()

        html_table = df.to_html(index=False)

        with open("static/dataframe.html", "w") as file:
            file.write(html_table)

    except Exception as e:
        print(f'Error generating plots: {e}')

def generate_summary(data):
    try:
        df = pd.json_normalize(data)

        df_filtered = df.loc[
            (df['average_response_time (hours)'] > 0) &
            (df['personalization_score'] > 0)
        ]
        
        num_contacts_user_initiated_true = int(df['user_initiated'].sum())
        average_response_time = df_filtered['average_response_time (hours)'].mean()
        total_emails_exchanged = int(df['emails_exchanged'].sum())
        average_sentiment_score = df['sentiment_score'].mean()
        average_personalization_score = df_filtered['personalization_score'].mean()
        average_emails_per_month = df['interaction_frequency (emails per month)'].mean()

        return {
            'num_initiations': num_contacts_user_initiated_true,
            'total_emails_exchanged': total_emails_exchanged,
            'avg_emails_per_month': round(average_emails_per_month, 2),
            'avg_response_time': round(average_response_time, 2),
            'avg_sentiment_score': round(average_sentiment_score, 2),
            'avg_personalization_score': round(average_personalization_score, 2)
        }

    except Exception as e:
        print(f'Error generating summary: {e}')
    
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    app.config['STATIC_FOLDER'] = STATIC_FOLDER

    upload_folder = app.config['UPLOAD_FOLDER']
    static_folder = app.config['STATIC_FOLDER']
    
    os.makedirs(upload_folder, exist_ok=True)
    os.makedirs(static_folder, exist_ok=True)

    file = request.files['mbox_file']
    user_email = request.form['email']

    if file:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{user_email}.mbox')
        file.save(file_path)

    nlp = spacy.load("en_core_web_sm")
    model_name = "mrm8488/bert-tiny-finetuned-sms-spam-detection"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    data = extract_mbox(file_path, model, tokenizer)
    
    try:
        obj = main(data, user_email, nlp)
        generate_plots(obj)
        summary = generate_summary(obj)
        return jsonify(summary), 200
    
    except Exception as e:
        print(f'An error occurred at the analyze endpoint: {e}')
        return jsonify({'error': 'An error occurred during analysis. Please try again.'}), 500

if __name__ == '__main__':
    app.run(debug=True)

# round during division, parse date, sorted()