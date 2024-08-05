import os
import re
import json
import spacy
import mailbox
import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from textblob import TextBlob
from collections import defaultdict, Counter
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
matplotlib.use('Agg') # Required for matplotlib to work with Flask

UPLOAD_FOLDER = 'uploads/'
STATIC_FOLDER = 'static/'

SPAM = {"jobs-listings@linkedin.com", "info@email.meetup.com", "team@mail.notion.so", "no-reply@messages.doordash.com", "updates-noreply@linkedin.com", "team@hiwellfound.com", "Starbucks@e.starbucks.com", "email@washingtonpost.com", "messages-noreply@linkedin.com", "rewards@e.starbucks.com", "info@meetup.com", "college@coll.herffjones.com", "venmo@email.venmo.com", "aws-marketing-email-replies@amazon.com", "chen.li@rexpandjob.com", "invitations@linkedin.com", "members@respond.kp.org", "no-reply@doordash.com", "bankofamerica@emcom.bankofamerica.com", "learn@itr.mail.codecademy.com", "noreply@alliance-mail.oa-bsa.org", "solatwestvillage@emailrelay.com", "alexanderqluong@gmail.com", "no-reply@modernmsg.com"}
INVITATION_KEYWORDS = {
    'invite', 'invites', 'invited', 'inviting', 'invitation', 'introduce', 'introduction', 'RSVP', 'like to meet', 'attend', 'event', 'participate'
}
ACCEPTANCE_KEYWORDS = {
    'accept', 'will attend'
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
    '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', 'am', 'pm'
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
    
    return None

def is_invitation(subject, body):
    try:
        subject = subject.lower() if subject else ''
        body = body.lower() if body else ''
        return any(keyword in subject or keyword in body for keyword in INVITATION_KEYWORDS)
    except Exception as e:
        print(f"Error in is_invitation: {e}")
        return False

def is_acceptance(subject, body):
    try:
        subject = subject.lower() if subject else ''
        body = body.lower() if body else ''
        return any(keyword in subject or keyword in body for keyword in ACCEPTANCE_KEYWORDS)
    except Exception as e:
        print(f"Error in is_acceptance: {e}")
        return False

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

def calculate_response_times(threads, response_times_by_contact, response_times, median_response_times, user_email):
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
                            response_times.append(response_time)
                            response_times_by_contact[contact].append(response_time)

        for contact, times in response_times_by_contact.items():
            if times:
                median_response_times[contact] = np.median(times)
                response_times_by_contact[contact] = sum(times) / len(times)

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
    try:
        user_initiation[address] = user_email in emails[0]['From']
    except Exception as e:
        print(f"Error in determining user initiation: {e}")

def calculate_personalization_score(personalization_scores, address, emails, user_email):
    try:
        nlp = spacy.load("en_core_web_sm")
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
        top_words = [word for word, count in most_common_words]
        keywords[address] = top_words
        
    except Exception as e:
        print(f"Error in finding keywords: {e}")

def process_message(message, email_content, contact_names, interaction_counts, invitation_counts, acceptance_counts, first_email_dates, last_email_dates, user_email):
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
            body = ' '

        if user_email not in from_address: # if the owner is not the sender, add the sender to the contact list
            split_address = from_address.split('<')
            if len(split_address) > 1: #  Two name-address formats: First Last <email> OR email
                name = split_address[0].strip()
                address = split_address[1].replace('>', '').strip()
                contacts.append(address)
                if address not in contact_names:
                    contact_names[address] = name
            else:
                address = from_address.strip()
                if address not in contact_names or contact_names[address] is None: # email only format; if not none, name is already known
                    contact_names[address] = None
        elif user_email in from_address: # not else because user can be part of a group email and we ignore this case
            split_to_addresses = to_addresses.split(',')
            if len(split_to_addresses) > 1: # multiple recipients
                for address in split_to_addresses:
                    if user_email not in address:
                        address = address.strip()
                        contacts.append(address)
                        if address not in contact_names or contact_names[address] is None: # multiple recipients format does not include names
                            contact_names[address] = None
            else:
                split_address = from_address.split('<')
                if len(split_address) > 1:
                    name = split_address[0].strip()
                    address = split_address[1].replace('>', '').strip()
                    if user_email not in address: # handle case where owner is the sender and recipient
                        contacts.append(address)
                        if address not in contact_names:
                            contact_names[address] = name
                else:
                    address = from_address.strip()
                    if address not in contact_names or contact_names[address] is None: # email only format; if not none, name is already known
                        contact_names[address] = None

        for address in contacts:
            email_content[address].append({
                'From': from_address,
                'To': to_addresses,
                'Subject': subject,
                'Date': date_str,
                'Body': body
            })

            interaction_counts[address] += 1

            if address in from_address and is_invitation(subject, body):
                invitation_counts[address] += 1
            
            # if user_email in from_address and is_acceptance(subject, body):
            #     acceptance_counts[address] += 1

            if date:
                if first_email_dates[address] is None or date < first_email_dates[address]:
                    first_email_dates[address] = date
                if last_email_dates[address] is None or date > last_email_dates[address]:
                    last_email_dates[address] = date

    except Exception as e:
        print(f"Error processing message: {e}")

def generate_tabular_data(email_content, contact_names, interaction_counts, invitation_counts, acceptance_counts, first_email_dates, last_email_dates, sentiment_scores, relationship_summaries, monthly_interactions, user_initiation, personalization_scores, follow_up_rates, keywords, response_times_by_contact, median_response_times, user_email):
    try:
        data = [{
            'contact': contact_names.get(contact, 'N/A'),
            'email_address': contact if contact else 'N/A',
            'emails_exchanged': interaction_counts.get(contact, 0),
            'number_of_invitations_received': invitation_counts.get(contact, 0),
            # 'number_of_accepted_invitations': acceptance_counts.get(contact, 0),
            'duration_known (months)': ((last_email_dates.get(contact) - first_email_dates.get(contact)).days) / 30 if first_email_dates.get(contact) and last_email_dates.get(contact) else 'N/A',
            'emails': email_content.get(contact, []),
            'sentiment_score': sentiment_scores.get(contact, 0),
            'relationship_summary': relationship_summaries.get(contact, 'N/A'),
            'user_initiated': user_initiation.get(contact, False),
            'interaction_frequency (emails per month)': monthly_interactions.get(contact, 0),
            'follow_up_rate (%)': follow_up_rates.get(contact, 0),
            'average_response_time (hours)': response_times_by_contact.get(contact, 0),
            'median_response_time (hours)': median_response_times.get(contact, 0),
            'keywords': keywords.get(contact, []),
            'personalization_score': personalization_scores.get(contact, 0) / (sum(1 for email in email_content.get(contact, []) if user_email in email['From']) if sum(1 for email in email_content.get(contact, []) if user_email in email['From']) > 0 else 1)
        } for contact in email_content]
        return data
    except Exception as e:
        print(f"Error generating tabular data: {e}")
        return []

def main(data, user_email):
    try:
        keywords = defaultdict(list)
        email_content = defaultdict(list)
        contact_names = defaultdict(str)
        sentiment_scores = defaultdict(str)   
        relationship_summaries = defaultdict(str)
        personalization_scores = defaultdict(int)
        response_times_by_contact = defaultdict(list)
        median_response_times = defaultdict(int)
        monthly_interactions = defaultdict(int)
        interaction_counts = defaultdict(int)
        invitation_counts = defaultdict(int)
        acceptance_counts = defaultdict(int)
        follow_up_rates = defaultdict(int)
        user_initiation = defaultdict(bool)
        last_email_dates = defaultdict(lambda: None)
        first_email_dates = defaultdict(lambda: None)
        threads = defaultdict(lambda: defaultdict(list))
        interactions_each_month = defaultdict(lambda: [0] * 13)
        response_times = []
        count = 0
        
        for entry in data: 
            process_message(entry, email_content, contact_names, interaction_counts, invitation_counts, acceptance_counts, first_email_dates, last_email_dates, user_email)

        for address, emails in email_content.items():
            # if address == 'contact_support.pzc@de(%)($)lica(%)($)rd.se' or address == 'CollegeBoard@noreply.collegeboard.org' or address == 'support_rk_royal@163.com':
            count += 1
            print(f'Processing contact {count} of {len(email_content)}')
            sentiment_analysis(sentiment_scores, relationship_summaries, address, emails)
            calculate_interaction_frequency(monthly_interactions, interactions_each_month, address, emails)
            calculate_personalization_score(personalization_scores, address, emails, user_email)
            find_keywords(keywords, address, emails)

            valid_emails = [email for email in emails if parse_date(email['Date']) is not None]
            sorted_emails = sorted(valid_emails, key=lambda x: parse_date(x['Date']))
            organize_by_thread(threads, address, sorted_emails)
            calculate_follow_up_rate(follow_up_rates, address, sorted_emails, user_email)
            user_initiated(user_initiation, address, sorted_emails, user_email)
        
        calculate_response_times(threads, response_times_by_contact, response_times, median_response_times, user_email)
        
        output_data = generate_tabular_data(
            email_content, contact_names, interaction_counts, invitation_counts, acceptance_counts,
            first_email_dates, last_email_dates, sentiment_scores, relationship_summaries, monthly_interactions,
            user_initiation, personalization_scores, follow_up_rates, keywords, response_times_by_contact, median_response_times, user_email
        )
        
        # pos = len([contact['sentiment_analysis'] for contact in output_data if contact['sentiment_analysis'] == 'positive'])
        # neutral = len([contact['sentiment_analysis'] for contact in output_data if contact['sentiment_analysis'] == 'neutral'])
        # neg = len([contact['sentiment_analysis'] for contact in output_data if contact['sentiment_analysis'] == 'negative'])

        # average_response_time = None
        # if response_times:
        #     average_response_time = sum(response_times) / len(response_times)

        # result = {
        #     'meeting_frequency': sum([contact['emails_exchanged'] for contact in output_data]) / len(output_data),
        #     'initiation_rate': sum([contact['user_initiated'] for contact in output_data]) / len(output_data),
        #     'number_of_unique_contacts': len(output_data),
        #     'calendar_utilization': sum([contact['number_of_invitations_received'] for contact in output_data]) / sum([contact['emails_exchanged'] for contact in output_data]),
        #     'average_email_length': sum([len(contact['emails']) for contact in output_data]) / len(output_data),
        #     'average_response_time': average_response_time,
        #     'email_volume': sum([contact['emails_exchanged'] for contact in output_data]),
        #     'engagement_rate': sum([contact['follow_up_rate'] for contact in output_data]) / len(output_data),
        #     'sentiment_analysis': sum([1 for contact in output_data if contact['sentiment_analysis'] == 'positive']) / len(output_data),
        #     'interaction_quality': f"Positive: {pos}, Neutral: {neutral}, Negative: {neg}"
        # }

        with open(f'{user_email}.json', 'w', encoding='utf-8') as json_file:
            json.dump(output_data, json_file, indent=4)

        print(f'Data has been written to {user_email}.json')

        return output_data

    except Exception as e:
        print(f'An error occurred: {e}')

def clean_headers(msg):
    allowed_headers = ['To', 'From', 'Subject', 'Date']
    cleaned_msg = {key: str(msg.get(key)) for key in allowed_headers if msg.get(key)}
    return cleaned_msg

def extract_mbox(input_mbox):
    processed_messages = []
    in_mbox = mailbox.mbox(input_mbox)
    count = 0

    for message in in_mbox:
        print(f"Extracting message #{count}")
        count += 1
        from_address = message.get('From', '')
        if from_address:
            if any(spam in from_address for spam in SPAM):
                continue
        
        if message.is_multipart():
            combined_payload = []
            try:
                for part in message.walk():
                    if part.get_content_type() == 'text/plain':
                        payload = part.get_payload(decode=True).decode('utf-8', errors='replace')
                        combined_payload.append(payload)
                
                combined_payload = '\n'.join(combined_payload)
                
                email_dict = {
                    'From': str(message.get('From')),
                    'To': str(message.get('To')),
                    'Subject': str(message.get('Subject')),
                    'Date': str(message.get('Date')),
                    'Body': combined_payload
                }

                processed_messages.append(email_dict)
            except Exception as e:
                print(f'Error processing message: {e}')
                continue

        else:
            if message.get_content_type() != 'text/plain':
                continue

            email_dict = clean_headers(message)
            email_dict['Body'] = message.get_payload(decode=True).decode('utf-8', errors='replace')
            with open('email.txt', 'a') as file:
                file.write(email_dict['Body'])
                file.write('=' * 100)
            processed_messages.append(email_dict)

    return processed_messages

def generate_plots(data):
    try:
        df = pd.DataFrame(data)

        # 1. Histogram of Average Response Times
        plt.figure(figsize=(12, 6))
        sns.histplot(df['average_response_time (hours)'], bins=100, kde=True, color='blue')
        plt.title('Histogram of Average Response Times')
        plt.xlabel('Average Response Time (hours)')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.savefig('static/histogram_response_times.png')
        plt.close()

        # 2. Histogram of Personalization Scores
        plt.figure(figsize=(12, 6))
        sns.histplot(df['personalization_score'], bins=100, kde=True, color='green')
        plt.title('Histogram of Personalization Scores')
        plt.xlabel('Personalization Score')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.savefig('static/histogram_personalization_scores.png')
        plt.close()

        # 3. Pie Chart of Relationship Summary
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
        
        # Filter out zero values for average_response_time and personalization_score
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
            'avg_emails_per_month': round(average_emails_per_month, 2) if not np.isnan(average_emails_per_month) else 0,
            'avg_response_time': round(average_response_time, 2) if not np.isnan(average_response_time) else 0,
            'avg_sentiment_score': round(average_sentiment_score, 2) if not np.isnan(average_sentiment_score) else 0,
            'avg_personalization_score': round(average_personalization_score, 2) if not np.isnan(average_personalization_score) else 0
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

    data = extract_mbox(file_path)
    
    try:
        obj = main(data, user_email)
        generate_plots(obj)
        summary = generate_summary(obj)
        return jsonify(summary), 200
    
    except Exception as e:
        print(f'An error occurred: {e}')
        return jsonify({'error': 'An error occurred during analysis. Please try again.'}), 500

if __name__ == '__main__':
    app.run(debug=True)

# emails displayed in chronological order, fix user_initiated, handle missing values, no need to assign dictionary values to None, round during division