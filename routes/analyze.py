import os
import re
import spacy
import torch
import mailbox
import pandas as pd
import torch.nn.functional as F
from datetime import datetime
from textblob import TextBlob
from supabase_config import client
from werkzeug.utils import secure_filename
from collections import defaultdict, Counter
from flask import render_template, request, jsonify, Blueprint, session
from transformers import AutoModelForSequenceClassification, AutoTokenizer

analyze = Blueprint('analyze', __name__)

supabase = client()

UPLOAD_FOLDER = 'uploads/'
CHUNK_FOLDER = os.path.join(UPLOAD_FOLDER, 'chunks')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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

def calculate_user_response_times(threads, user_response_times_by_contact, user_email):
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

def calculate_thread_length(threads, thread_lengths):
    try:
        for contact, threadz in threads.items():
            for emails in threadz.values():
                thread_lengths[contact] += 1 + len(emails)
            thread_lengths[contact] /= len(threadz)
    except Exception as e:
        print(f"Error in calculating thread length: {e}")

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

def assign_tier(contact, last_email_dates, relationship_summaries):
    current_date = pd.Timestamp(datetime.now()).tz_localize('UTC')
    six_months_ago = current_date - pd.DateOffset(months=6)
    last_contact_str = last_email_dates.get(contact, None)
    sentiment = relationship_summaries.get(contact, 'N/A')
    
    if not last_contact_str or sentiment == 'N/A':
        return 0

    try:
        last_contact = pd.to_datetime(last_contact_str, utc=True)
    except Exception as e:
        print(f"Error parsing date for contact {contact}: {e}")
        return 0

    active = six_months_ago <= last_contact
    
    if active and sentiment == 'positive':
        return 1
    elif not active and sentiment == 'positive':
        return 2
    elif active and sentiment == 'neutral':
        return 3
    else:
        return 4

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

def generate_tabular_data(email_meta, email_content, contact_names, interaction_counts, invitation_counts, first_email_dates, last_email_dates, sentiment_scores, relationship_summaries, monthly_interactions, user_initiation, personalization_scores, follow_up_rates, keywords, contact_response_times, user_response_times_by_contact, user_email, thread_lengths, account_id):
    try:
        data = [{
            'tier': assign_tier(contact, last_email_dates, relationship_summaries),
            'contact': contact_names.get(contact, 'N/A'),
            'email_address': contact if contact else 'N/A',
            'emails_exchanged': interaction_counts.get(contact, 0),
            'invitations_received': invitation_counts.get(contact, 0),
            'months_known': round(((last_email_dates.get(contact) - first_email_dates.get(contact)).days) / 30, 2) if first_email_dates.get(contact) and last_email_dates.get(contact) else 0,
            'last_date_of_contact': str(last_email_dates.get(contact, 'N/A')),
            'sentiment_score': round(sentiment_scores.get(contact, 0), 2),
            'relationship_summary': relationship_summaries.get(contact, 'N/A'),
            'user_initiated': user_initiation.get(contact, False),
            'emails_per_month': int(monthly_interactions.get(contact, 0)),
            'follow_up_rate_percentage': int(follow_up_rates.get(contact, 0)),
            "contact_avg_response_time_hours": round(contact_response_times.get(contact, 0), 2),
            'user_avg_response_time_hours': round(user_response_times_by_contact.get(contact, 0), 2),
            'average_email_thread_length': int(thread_lengths.get(contact, 0)),
            'keywords': keywords.get(contact, 'N/A'),
            'personalization_score': round(personalization_scores.get(contact, 0) / (sum(1 for email in email_meta.get(contact, []) if user_email in email['From']) or 1), 2),
            'account_id': account_id

        } for contact in email_content]

        response = (
            supabase.table("analysis")
            .upsert(
                data,
                on_conflict="email_address, account_id"
            )
            .execute()
        )

    except Exception as e:
        print(f"Error generating tabular data: {e}")

def main(data, user_email, nlp, account_id):
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
        monthly_interactions = defaultdict(int)
        interaction_counts = defaultdict(int)
        invitation_counts = defaultdict(int)
        follow_up_rates = defaultdict(int)
        thread_lengths = defaultdict(int)
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
        
        calculate_user_response_times(threads, user_response_times_by_contact, user_email)
        calculate_contact_response_times(threads, contact_response_times, user_email)
        calculate_thread_length(threads, thread_lengths)
        
        generate_tabular_data(
            email_meta, email_content, contact_names, interaction_counts, invitation_counts,
            first_email_dates, last_email_dates, sentiment_scores, relationship_summaries, monthly_interactions, user_initiation,
            personalization_scores, follow_up_rates, keywords, contact_response_times, user_response_times_by_contact, user_email, thread_lengths, account_id
        )

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
            count += 1
            print(f"Extracting message #{count} of {len(in_mbox)}")

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

def handle_chunk_upload():
    chunk_id = request.form.get('chunkId')
    filename = request.form.get('filename')
    file = request.files['file']

    file_path = os.path.join(UPLOAD_FOLDER, filename)

    if not os.path.exists(file_path):
        with open(file_path, 'wb') as f:
            f.write(file.read())
    else:
        with open(file_path, 'ab') as f:
            f.write(file.read())

@analyze.route('/upload-chunk', methods=['POST'])
def upload_chunk():
    file = request.files['file']
    chunk_index = int(request.form['chunkIndex'])
    total_chunks = int(request.form['totalChunks'])
    file_id = request.form['fileId']

    chunk_dir = os.path.join(CHUNK_FOLDER, file_id)
    os.makedirs(chunk_dir, exist_ok=True)

    chunk_file = os.path.join(chunk_dir, f'chunk_{chunk_index}')
    file.save(chunk_file)

    if chunk_index == total_chunks - 1:
        filename = secure_filename(file_id)
        with open(os.path.join(UPLOAD_FOLDER, filename), 'wb') as outfile:
            for i in range(total_chunks):
                chunk_file = os.path.join(chunk_dir, f'chunk_{i}')
                with open(chunk_file, 'rb') as infile:
                    outfile.write(infile.read())
                os.remove(chunk_file)
        os.rmdir(chunk_dir)

    return jsonify({'success': True})

@analyze.route('/', methods=['GET', 'POST'])
def process_inbox():
    supabase = client()
    admin_id = session.get('user_id')
    if request.method == 'POST':
        # op = request.form.get('op')
        # if op == 'chunk_upload':
        #     handle_chunk_upload()
        #     return jsonify({'success': 'Chunk uploaded successfully'}), 200
            
        # file_path = os.path.join(UPLOAD_FOLDER, f'{request.form["filename"]}')
        data = request.get_json()
        file_id = data.get('file_id')
        file_path = os.path.join(UPLOAD_FOLDER, f'{file_id}')
        user_email = data.get('email')

        response = (
            supabase.table("accounts")
            .select("id")
            .eq("email", user_email)
            .eq("admin_id", admin_id)
            .execute()
        )

        if not response.data:
            (supabase.table("accounts")
            .insert({"email": user_email, "admin_id": admin_id})
            .execute()
            )
        
        query = (
            supabase.table("accounts")
            .select("id")
            .eq("email", user_email)
            .eq("admin_id", admin_id)
            .execute()
        )

        account_id = query.data[0]['id']
        
        nlp = spacy.load("en_core_web_sm")
        model_name = "mrm8488/bert-tiny-finetuned-sms-spam-detection"
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        data = extract_mbox(file_path, model, tokenizer)
        
        try:
            main(data, user_email, nlp, account_id)
            return jsonify({"success": "Successfully uploaded inbox"}), 200
    
        except Exception as e:
            print(f'An error occurred at the analyze endpoint: {e}')
            return jsonify({'error': 'An error occurred during analysis. Please try again.'}), 500
        

    query = (
        supabase.table("accounts")
        .select("email", "id")
        .eq("admin_id", admin_id)
        .execute()
    )

    accounts = query.data
    
    return render_template('upload.html', accounts=accounts)