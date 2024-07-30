import os
import json
import spacy
import mailbox
import pandas as pd
from groq import Groq
from datetime import datetime
from dotenv import load_dotenv
from collections import defaultdict, Counter

load_dotenv()

OWNER_NAME = 'User'
OWNER_EMAIL = 'user@example.com'

INVITATION_KEYWORDS = INVITATION_KEYWORDS = {
    'invite', 'invites', 'invited', 'inviting', 'invitation', 'introduce', 'introduction', 'RSVP', 'like to meet', 'attend', 'event', 'participate'
}
ACCEPTANCE_KEYWORDS = {
    'accept', 'confirm', 'will attend', 'agree'
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
        return datetime.strptime(date_str, '%a, %d %b %Y %H:%M:%S %z')
    except ValueError:
        return None

def is_invitation(subject, body):
    subject = subject.lower() if subject else ''
    body = body.lower() if body else ''
    return any(keyword in subject or keyword in body for keyword in INVITATION_KEYWORDS)

def is_acceptance(subject, body):
    subject = subject.lower() if subject else ''
    body = body.lower() if body else ''
    return any(keyword in subject or keyword in body for keyword in ACCEPTANCE_KEYWORDS)

def groq_general_sentiment(feed):

    chat_completion = client.chat.completions.create(
    model="llama3-70b-8192",
    messages=[
        {
            "role": "system",
            "content": "you are a sentiment analysis expert"
        },
        {
            "role": "user",
            "content":
            """
            Conduct sentiment analysis on the provided email content.
            The analysis must result in a single word indicating the sender's sentiment, chosen exclusively from \"positive,\" \"negative,\" or \"neutral.\"
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

# summarize one sentence relationship and add it into last column
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
            "content": 
            """
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

def sentiment_analysis(sentiment_analyses, summary_relationships, contact, emails):
    feed = " ".join([email['Body'] for email in emails])
    sentiment_analysis = groq_general_sentiment(feed)
    sentiment_analyses[contact] = sentiment_analysis
    summary_relationship = groq_summary_relationship(feed)
    summary_relationships[contact] = summary_relationship

def calculate_response_time(threads, response_times, emails):

    # Group emails into threads based on the subject
    for email in emails:
        subject = email['Subject']
        if subject and ('Re:' in subject or 'RE' in subject):
            threads[subject].append(email)

    # Calculate response times for each thread
    for thread_emails in threads.values():
        # Sort emails by date
        sorted_emails = sorted(thread_emails, key=lambda x: parse_date(x['Date']) or datetime.min)
        for i in range(1, len(sorted_emails)):
            current_email = sorted_emails[i]
            previous_email = sorted_emails[i - 1]
            
            # Check if the current email is from the owner and the previous one is not
            if OWNER_EMAIL in current_email['From'] and OWNER_EMAIL not in previous_email['From']:
                response_time = (parse_date(current_email['Date']) - parse_date(previous_email['Date'])).seconds
                response_time /= 3600
                response_times.append(response_time)

    if response_times:
        average_response_time = sum(response_times) / len(response_times)
        return average_response_time
    else:
        return None
    
def calculate_follow_up_rate(follow_up, contact, emails):
    from_contact_email_count = sum(1 for email in emails if contact in email['From'])
    follow_up_count = 0

    for i in range(1, len(emails)):
        current_email = emails[i]
        previous_email = emails[i - 1]

        if OWNER_EMAIL in current_email['From'] and OWNER_EMAIL not in previous_email['From']:
            follow_up_count += 1

    follow_up[contact] = follow_up_count / from_contact_email_count * 100

    return follow_up
    
# counts the average number of interactions per month for months containing at least one email
def interaction_frequency(monthly_interactions, email_counts, contact, emails):
    for email in emails:
        date_str = email["Date"]
        date_obj = datetime.strptime(date_str, '%a, %d %b %Y %H:%M:%S %z')
        month = date_obj.strftime('%m')
        month = int(month.lstrip('0'))
        email_counts[contact][month] += 1

    for contacts, values in email_counts.items():
        total_sum = 0
        non_zero_months = 0
        for value in values:
            if value != 0:
                total_sum += value
                non_zero_months += 1
        monthly_interactions[contacts] = total_sum / non_zero_months
            
    return monthly_interactions

def user_initiated(user_initiation, contact, emails):
    if OWNER_EMAIL in emails[0]['From']:
        user_initiation[contact] = True
    else:
        user_initiation[contact] = False

def calculate_personalization_score(personalization_scores, contact, emails):
    nlp = spacy.load("en_core_web_sm")
    for email in emails:
        doc = nlp(email['Body'])
        for ent in doc.ents:
            personalization_scores[contact] += 1

def find_keywords(keywords, contact, emails):
    all_messages = " ".join([email['Body'] for email in emails])
    all_messages = all_messages.lower()
    all_messages = all_messages.replace('\n', ' ')
    all_messages = all_messages.replace('.', '')
    all_messages = all_messages.replace(',', '')
    all_messages = all_messages.split()
    filtered_messages = [word for word in all_messages if word not in STOP_WORDS and contact.split()[0].lower() not in word and OWNER_NAME.lower() not in word]
    word_counts = Counter(filtered_messages)
    most_common_words = word_counts.most_common(5)
    keywords[contact] = most_common_words

    return keywords

def process_message(message, email_content, email_addresses, interaction_counts, invitation_counts, acceptance_counts, first_email_dates, last_email_dates):
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

def prepare_output(email_content, email_addresses, interaction_counts, invitation_counts, acceptance_counts, first_email_dates, last_email_dates, sentiment_analyses, summary_relationships, monthly_interactions, user_initiation, personalization_scores, follow_up_rates, keywords):
    return [{
                'contact': contact,
                'email_address': email_addresses[contact],
                'number_of_invitations': invitation_counts[contact],
                'number_of_accepted_invitations': acceptance_counts[contact],
                'duration_known': (last_email_dates[contact] - first_email_dates[contact]).days if first_email_dates[contact] and last_email_dates[contact] else None,
                'number_of_invitations': invitation_counts[contact],
                'email_rate': (interaction_counts[contact] / ((last_email_dates[contact] - first_email_dates[contact]).days + 1)) if first_email_dates[contact] and last_email_dates[contact] else None,
                'emails': email_content[contact],
                'sentiment_analysis': sentiment_analyses[contact],
                'summary_relationship': summary_relationships[contact],
                'monthly_interactions': monthly_interactions[contact],
                'user_initiated': user_initiation[contact],
                'personalization_scores': personalization_scores[contact],
                'follow_up_rate': follow_up_rates[contact],
                'keywords': keywords[contact]
            }
            for contact in email_content
            ]
    
    # df = pd.DataFrame(data)
    # print(df)

def main():
    try:
        mbox = mailbox.mbox('dev.mbox')

        threads = defaultdict(list)
        keywords = defaultdict(list)
        email_content = defaultdict(list)
        email_addresses = defaultdict(str)
        sentiment_analyses = defaultdict(str)
        summary_relationships = defaultdict(str)
        personalization_scores = defaultdict(int)
        monthly_interactions = defaultdict(int)
        interaction_counts = defaultdict(int)
        invitation_counts = defaultdict(int)
        acceptance_counts = defaultdict(int)
        follow_up_rates = defaultdict(int)
        user_initiation = defaultdict(bool)
        last_email_dates = defaultdict(lambda: None)
        first_email_dates = defaultdict(lambda: None)
        email_counts = defaultdict(lambda: [0] * 13)
        response_times = []

        for message in mbox:
            process_message(message, email_content, email_addresses, interaction_counts, invitation_counts, acceptance_counts, first_email_dates, last_email_dates)

        for contact, emails in email_content.items():
            emails = sorted(emails, key=lambda x: parse_date(x['Date']) or datetime.min)
            # sentiment_analysis(sentiment_analyses, summary_relationships, contact, emails)
            calculate_response_time(threads, response_times, emails)
            interaction_frequency(monthly_interactions, email_counts, contact, emails)
            user_initiated(user_initiation, contact, emails)
            calculate_personalization_score(personalization_scores, contact, emails)
            calculate_follow_up_rate(follow_up_rates, contact, emails)
            find_keywords(keywords, contact, emails)
        
        output_data = prepare_output(email_content, email_addresses, interaction_counts, invitation_counts, acceptance_counts, first_email_dates, last_email_dates,
                                     sentiment_analyses, summary_relationships, monthly_interactions, user_initiation, personalization_scores, follow_up_rates, keywords)

        with open('email_data.json', 'w', encoding='utf-8') as json_file:
            json.dump(output_data, json_file, indent=4)

        print('Data has been written to email_data.json')

    except Exception as e:
        print(f'An error occurred: {e}')

if __name__ == '__main__':
    main()