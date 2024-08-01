import os
import json
import spacy
import mailbox
import numpy as np
import pandas as pd
from groq import Groq
from datetime import datetime
from dotenv import load_dotenv
from collections import defaultdict, Counter

load_dotenv()

OWNER_NAME = 'Evan Tan'
OWNER_EMAIL = 'evant5252@gmail.com'

INVITATION_KEYWORDS = INVITATION_KEYWORDS = {
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

client = Groq(
    api_key = os.getenv("GROQ_API_KEY"),
)

def parse_date(date_str):
    try:
        # Remove any extra text after the timezone offset
        clean_date_str = date_str.split(' (')[0]
        return datetime.strptime(clean_date_str, '%a, %d %b %Y %H:%M:%S %z')
    except ValueError as e:
        print(f"Error parsing date: {e}")
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

def groq_general_sentiment(feed):
    try:
        chat_completion = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {
                    "role": "system",
                    "content": "you are a sentiment analysis expert"
                },
                {
                    "role": "user",
                    "content": """
                        Conduct sentiment analysis on the provided email content.
                        The analysis must result in a single word indicating the sender's sentiment, chosen exclusively from "positive," "negative," or "neutral."
                        Please return only the selected word.
                    """
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
    except Exception as e:
        print(f"Error in groq_general_sentiment: {e}")
        return "unknown"

def groq_summary_relationship(feed):
    try:
        chat_completion = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {
                    "role": "system",
                    "content": "you are a sentiment analysis expert"
                },
                {
                    "role": "user",
                    "content": """
                        Summarize my relationship with this contact based on provided email content in just one sentence.
                        Don't start with Here is a summary of your relationship with this contact in one sentence:
                    """
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
    except Exception as e:
        print(f"Error in groq_summary_relationship: {e}")
        return "unknown"

def sentiment_analysis(sentiment_analyses, summary_relationships, contact, emails):
    try:
        feed = " ".join([email['Body'] for email in emails])
        sentiment_analysis = groq_general_sentiment(feed)
        sentiment_analyses[contact] = sentiment_analysis
        summary_relationship = groq_summary_relationship(feed)
        summary_relationships[contact] = summary_relationship
    except Exception as e:
        print(f"Error in sentiment_analysis: {e}")

def organize_by_thread(threads, contact, emails):
    try:
        for email in emails:
            subject = email['Subject']
            if subject and ('Re:' in subject or 'RE' in subject):
                if contact not in threads:
                    threads[contact] = {}
                if subject not in threads[contact]:
                    threads[contact][subject] = []
                threads[contact][subject].append(email)
    except Exception as e:
        print(f"Error in organize_by_thread: {e}")

def calculate_response_times(threads, response_times_by_contact, response_times, median_response_times):
    try:
        for contact, emails in threads.items():
            for email_thread in emails.values():
                for i in range(1, len(email_thread)):
                    current_email = email_thread[i]
                    previous_email = email_thread[i - 1]
                    
                    # Check if the current email is from the owner and the previous one is not
                    if OWNER_EMAIL in current_email['From'] and OWNER_EMAIL not in previous_email['From']:
                        current_date = parse_date(current_email['Date'])
                        previous_date = parse_date(previous_email['Date'])
                        if current_date and previous_date:
                            response_time = (current_date - previous_date).seconds
                            response_time /= 3600
                            response_times.append(response_time)
                            response_times_by_contact.setdefault(contact, []).append(response_time)

        for contact, times in response_times_by_contact.items():
            if times:
                median_response_times[contact] = np.median(times)
                response_times_by_contact[contact] = sum(times) / len(times)
            else:
                response_times_by_contact[contact] = None

    except Exception as e:
        print(f"Error in calculating response times: {e}")

def calculate_follow_up_rate(follow_up, contact, emails):
    try:
        from_contact_email_count = sum(1 for email in emails if contact in email['From'])
        follow_up_count = 0

        for i in range(1, len(emails)):
            current_email = emails[i]
            previous_email = emails[i - 1]

            if OWNER_EMAIL in current_email['From'] and OWNER_EMAIL not in previous_email['From']:
                follow_up_count += 1

        follow_up[contact] = (follow_up_count / from_contact_email_count * 100) if from_contact_email_count > 0 else 0
    except Exception as e:
        print(f"Error in calculating follow up rate: {e}")

def interaction_frequency(monthly_interactions, email_counts, contact, emails):
    try:
        for email in emails:
            date_str = email["Date"]
            try:
                date_obj = datetime.strptime(date_str, '%a, %d %b %Y %H:%M:%S %z')
                month = date_obj.strftime('%m')
                month = int(month.lstrip('0'))
                if contact not in email_counts:
                    email_counts[contact] = {}
                if month not in email_counts[contact]:
                    email_counts[contact][month] = 0
                email_counts[contact][month] += 1
            except ValueError as e:
                print(f"Error parsing date in interaction_frequency: {e}")

        for contact, values in email_counts.items():
            total_sum = 0
            non_zero_months = 0
            for value in values.values():
                if value != 0:
                    total_sum += value
                    non_zero_months += 1
            monthly_interactions[contact] = total_sum / non_zero_months if non_zero_months > 0 else 0
    except Exception as e:
        print(f"Error in calculating interaction frequency: {e}")

def user_initiated(user_initiation, contact, emails):
    try:
        user_initiation[contact] = OWNER_EMAIL in emails[0]['From']
    except Exception as e:
        print(f"Error in determining user initiation: {e}")

def calculate_personalization_score(personalization_scores, contact, emails):
    try:
        nlp = spacy.load("en_core_web_sm")
        if contact not in personalization_scores:
            personalization_scores[contact] = 0
        for email in emails:
            doc = nlp(email['Body'])
            for ent in doc.ents:
                personalization_scores[contact] += 1
    except Exception as e:
        print(f"Error in calculating personalization score: {e}")

def find_keywords(keywords, contact, emails):
    try:
        all_messages = " ".join([email['Body'] for email in emails])
        all_messages = all_messages.lower()
        all_messages = all_messages.replace('\n', ' ')
        all_messages = all_messages.replace('.', '')
        all_messages = all_messages.replace(',', '')
        all_messages = all_messages.split()
        filtered_messages = [word for word in all_messages if word not in STOP_WORDS and contact.split()[0].lower() not in word and OWNER_NAME.lower() not in word]
        word_counts = Counter(filtered_messages)
        most_common_words = word_counts.most_common(5)
        top_words = [word for word, count in most_common_words]
        keywords[contact] = top_words
    except Exception as e:
        print(f"Error in finding keywords: {e}")

import mailbox
import json
from collections import defaultdict
from datetime import datetime

def process_message(message, email_content, email_addresses, interaction_counts, invitation_counts, acceptance_counts, first_email_dates, last_email_dates):
    try:
        from_address = message['From']
        to_addresses = message.get_all('To', [])
        date_str = message['Date']
        date = parse_date(date_str)
        subject = message['Subject']
        body = message.get_payload()
        contacts = []

        if from_address and OWNER_EMAIL not in from_address:
            contact = from_address.split('<')[0].strip()
            contacts.append(contact)
            email = from_address.split('<')[1].replace('>', '').strip()
            if contact not in email_addresses:
                email_addresses[contact] = email
        else:
            for address in to_addresses:
                address = address.strip()
                if OWNER_EMAIL not in address:
                    contact = address.split('<')[0].strip()
                    contacts.append(contact)
                    email = address.split('<')[1].replace('>', '').strip()
                    if contact not in email_addresses:
                        email_addresses[contact] = email

        for contact in contacts:
            email_content[contact].append({
                'From': from_address,
                'To': to_addresses,
                'Subject': subject,
                'Date': date_str,
                'Body': body
            })
            interaction_counts[contact] += 1

            if OWNER_EMAIL not in from_address and is_invitation(subject, body):
                invitation_counts[contact] += 1
            
            if OWNER_EMAIL in from_address and is_acceptance(subject, body):
                acceptance_counts[contact] += 1

            if contact == from_address and is_invitation(subject, body):
                invitation_counts[contact] += 1
            if date:
                if first_email_dates[contact] is None or date < first_email_dates[contact]:
                    first_email_dates[contact] = date
                if last_email_dates[contact] is None or date > last_email_dates[contact]:
                    last_email_dates[contact] = date

    except Exception as e:
        print(f"Error processing message: {e}")

def generate_tabular_data(email_content, email_addresses, interaction_counts, invitation_counts, acceptance_counts, first_email_dates, last_email_dates, sentiment_analyses, summary_relationships, monthly_interactions, user_initiation, personalization_scores, follow_up_rates, keywords, response_times_by_contact, median_response_times):
    try:
        data = [{
            'contact': contact,
            'email_address': email_addresses.get(contact, 'N/A'),
            'emails_exchanged': interaction_counts.get(contact, 0),
            'number_of_invitations_received': invitation_counts.get(contact, 0),
            'number_of_accepted_invitations': acceptance_counts.get(contact, 0),
            'duration_known (months)': ((last_email_dates.get(contact, datetime.min) - first_email_dates.get(contact, datetime.min)).days) / 30 if first_email_dates.get(contact) and last_email_dates.get(contact) else None,
            'emails': email_content.get(contact, []),
            'sentiment_analysis': sentiment_analyses.get(contact, 'N/A'),
            'summary_relationship': summary_relationships.get(contact, 'N/A'),
            'user_initiated': user_initiation.get(contact, False),
            'interaction_frequency (emails per month)': monthly_interactions.get(contact, 0),
            'follow_up_rate': follow_up_rates.get(contact, 0),
            'average_response_time (hours)': response_times_by_contact.get(contact, 'Not enough data'),
            'median_response_time (hours)': median_response_times.get(contact, 'Not enough data'),
            'keywords': keywords.get(contact, []),
            'personalization_score': personalization_scores.get(contact, 0),
        } for contact in email_content]
        return data
    except Exception as e:
        print(f"Error generating tabular data: {e}")
        return []

def main():
    try:
        mbox = mailbox.mbox('my.mbox')

        keywords = defaultdict(list)
        email_content = defaultdict(list)
        email_addresses = defaultdict(str)
        sentiment_analyses = defaultdict(str)   
        summary_relationships = defaultdict(str)
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
        email_counts = defaultdict(lambda: [0] * 13)
        response_times = []
        count = 0

        with open('part.txt', 'w') as file:
            pass

        for message in mbox:

            if message.is_multipart():
                while count < 25:
                    count += 1
                    for part in message.iter_parts():
                        with open('part.txt', 'a') as f:
                            f.write(str(part))
                            f.write('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
                        if part.get_content_type() == 'text/plain':
                            process_message(part, email_content, email_addresses, interaction_counts, invitation_counts, acceptance_counts, first_email_dates, last_email_dates)

            if message.get_content_type() != 'text/plain':
                continue
        
            if not all([
                message.get('From'),
                message.get('To'),
                message.get('Subject'),
                message.get('Date'),
                message.get('Body')
            ]): continue

            process_message(message, email_content, email_addresses, interaction_counts, invitation_counts, acceptance_counts, first_email_dates, last_email_dates)

        for contact, emails in email_content.items():
            emails = sorted(emails, key=lambda x: parse_date(x['Date']) or datetime.min)
            # sentiment_analysis(sentiment_analyses, summary_relationships, contact, emails)
            organize_by_thread(threads, contact, emails)
            interaction_frequency(monthly_interactions, email_counts, contact, emails)
            user_initiated(user_initiation, contact, emails)
            calculate_personalization_score(personalization_scores, contact, emails)
            calculate_follow_up_rate(follow_up_rates, contact, emails)
            find_keywords(keywords, contact, emails)
        
        calculate_response_times(threads, response_times_by_contact, response_times, median_response_times)
        
        output_data = generate_tabular_data(
            email_content, email_addresses, interaction_counts, invitation_counts, acceptance_counts,
            first_email_dates, last_email_dates, sentiment_analyses, summary_relationships, monthly_interactions,
            user_initiation, personalization_scores, follow_up_rates, keywords, response_times_by_contact, median_response_times
        )
        
        # pos = len([contact['sentiment_analysis'] for contact in output_data if contact['sentiment_analysis'] == 'positive'])
        # neutral = len([contact['sentiment_analysis'] for contact in output_data if contact['sentiment_analysis'] == 'neutral'])
        # neg = len([contact['sentiment_analysis'] for contact in output_data if contact['sentiment_analysis'] == 'negative'])

        # average_response_time = None
        # if response_times:
        #     average_response_time = sum(response_times) / len(response_times)

        # result = {
        #     'meeting_frequency': sum([contact['emails_exchanged'] for contact in output_data]) / len(output_data), TODO
        #     'initiation_rate': sum([contact['user_initiated'] for contact in output_data]) / len(output_data),
        #     'number_of_unique_contacts': len(output_data),
        #     'calendar_utilization': sum([contact['number_of_invitations_received'] for contact in output_data]) / sum([contact['emails_exchanged'] for contact in output_data]), TODO
        #     'average_email_length': sum([len(contact['emails']) for contact in output_data]) / len(output_data), TODO
        #     'average_response_time': average_response_time,
        #     'email_volume': sum([contact['emails_exchanged'] for contact in output_data]),
        #     'engagement_rate': sum([contact['follow_up_rate'] for contact in output_data]) / len(output_data),
        #     'sentiment_analysis': sum([1 for contact in output_data if contact['sentiment_analysis'] == 'positive']) / len(output_data), TODO
        #     'interaction_quality': f"Positive: {pos}, Neutral: {neutral}, Negative: {neg}"
        # }

        with open('tabular_data.json', 'w', encoding='utf-8') as json_file:
            json.dump(output_data, json_file, indent=4)

        # with open('general_data.json', 'w', encoding='utf-8') as json_file:
        #     json.dump(result, json_file, indent=4)

        print('Data has been written to tabular_data.json')
        # print('General data has been written to general_data.json')

    except Exception as e:
        print(f'An error occurred: {e}')

# if __name__ == '__main__':
main()

# filter marketing emails, emails displayed in chronological order
# filter mbox to remove unnecessary headers