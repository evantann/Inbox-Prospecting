import re
import os
import mailbox
import pandas as pd
from groq import Groq
from datetime import datetime
from dotenv import load_dotenv
from collections import defaultdict
from email.utils import parsedate_tz, mktime_tz

load_dotenv()

# extract emails from mbox file and store them into a list of dictionary
def extract_emails(mbox_file):
    mbox = mailbox.mbox(mbox_file)
    emails = []
    for message in mbox:
        if message.is_multipart():
            for part in message.walk():
                if part.get_content_type() == 'text/plain':
                    body = part.get_payload(decode=True).decode('utf-8', errors='ignore')
                    break
        else:
            body = message.get_payload(decode=True).decode('utf-8', errors='ignore')
                
        emails.append({
            'Subject': message['Subject'],
            'From': message['From'],
            'To': message['To'],
            'Date': message['Date'],
            'Body': body
        })

    return emails

# filter emails to exclude marketing email and non-personal email
# Exclude contacts to whom you have sent less than 3 emails in your entire history. 
def filter_emails(emails):
    filtered_emails = []
    for email in emails:
        if is_marketing_email(email['Subject'], email['From']):
            continue
        
        if is_non_personal_email(email['Subject'], email['From']):
            continue
        
        if email['To'] in close_contacts(emails) or email['From'] in close_contacts(emails):
            filtered_emails.append(email)

    return filtered_emails

# check if it's marketing and promotional email using common marketing keyword filters and sender domain
def is_marketing_email(subject, from_address):
    if subject is None or from_address is None:
        return False
    
    if not isinstance(subject, str):
        subject = str(subject)
    if not isinstance(from_address, str):
        from_address = str(from_address)

    marketing_keywords = ["sale", "discount", "save", "off", "offer", "deal", "clearance",
        "free", "giveaway", "promo", "bogo", "limited time", "hurry",
        "act now", "last chance", "exclusive", "only", "final", "ending soon",
        "while supplies last", "today only", "new", "launch", "event",
        "introducing", "announcement", "coming soon", "just arrived",
        "premiere", "grand opening", "subscribe", "member", "loyalty", "join",
        "insider", "vip", "rewards", "holiday", "black friday", "cyber monday",
        "christmas", "new year", "summer", "winter", "spring", "fall",
        "halloween", "thanksgiving", "valentine’s day", "easter", "buy now",
        "shop", "click", "get", "check out", "download", "rsvp",
        "sign up", "order", "price drop", "markdown", "special offer",
        "limited offer", "bundle", "package", "combo", "free shipping",
        "fast delivery", "delivered", "shipping now"
        ]
    marketing_domains = ["@newsletter", "@promotions", "@news", "@updates", "@deals", "@offers",
        "@sales", "@specials", "@amazon.com", "@ebay.com", "@walmart.com",
        "@target.com", "@bestbuy.com", "@homedepot.com", "@lowes.com",
        "@macys.com", "@kohls.com", "@gap.com", "@oldnavy.com", "@zara.com",
        "@hm.com", "@forever21.com", "@netflix.com", "@hulu.com",
        "@spotify.com", "@pandora.com", "@apple.com", "@disneyplus.com",
        "@booking.com", "@expedia.com", "@airbnb.com", "@marriott.com",
        "@hilton.com", "@tripadvisor.com", "@starbucks.com", "@mcdonalds.com",
        "@ubereats.com", "@doordash.com", "@grubhub.com", "@dell.com",
        "@hp.com", "@lenovo.com", "@microsoft.com", "@intel.com", "@nvidia.com",
        "@walgreens.com", "@cvs.com", "@riteaid.com", "@gnc.com", "@nike.com",
        "@adidas.com", "@underarmour.com", "@reebok.com", "@sephora.com",
        "@ulta.com", "@bathandbodyworks.com", "@nytimes.com", "@wsj.com",
        "@washingtonpost.com"
        ]
    
    if any(keyword in subject.lower() for keyword in marketing_keywords):
        return True
    
    if any(domain in from_address.lower() for domain in marketing_domains):
        return True
    
    return False

# check if an email is automatically sent from system, instead of from a personal one
def is_non_personal_email(subject, from_address):
    if subject is None or from_address is None:
        return False
    
    non_personal_keywords = ["notification", "alert", "update", "reminder", "confirmation", "receipt",
        "invoice", "statement", "report", "summary", "no-reply", "noreply",
        "do-not-reply", "donotreply", "system", "admin", "automated", "auto-reply",
        "autoreply", "support", "helpdesk", "service", "account", "order", "purchase",
        "transaction", "booking", "reservation", "payment", "billing", "subscription",
        "security", "compliance", "verification", "password", "login", "access"
        ]
    
    if any(keyword in subject for keyword in non_personal_keywords):
        return True

    if any(keyword in from_address for keyword in non_personal_keywords):
        return True
    
    return False

# find contacts with whom user had sent at least 3 messages 
def close_contacts(emails):
    close_contacts_email = []
    contact_list = unique_contacts(emails)
    for contact, count in contact_list.items():
        if count >= 3:
            close_contacts_email.append(contact)
    return close_contacts_email

def response_time(emails):
    threads = defaultdict(list)

    for email in emails:
        subject = email['Subject']
        if 'RE' in subject or 'Re' in subject:
            threads[subject].append(email)

    response_time = calculate_response_times(threads)
    return response_time

def calculate_response_times(email_dict):
    response_times = []
    result = 0

    for key, emails_list in email_dict.items():
        others_emails = [email for email in emails_list if my_email not in email['From']]
        user_emails = [email for email in emails_list if  my_email in email['From']]

        others_emails.sort(key=lambda x: parse_date(x['Date']))
        user_emails.sort(key=lambda x: parse_date(x['Date']))

        user_idx = 0
        for other_email in others_emails:
            other_time = parse_date(other_email['Date'])
            
            while user_idx < len(user_emails):
                user_email = user_emails[user_idx]
                user_time = parse_date(user_email['Date'])
                if user_time > other_time:
                    response_time = (user_time - other_time) / 3600
                    response_times.append(response_time)
                    user_idx += 1
                    break
                user_idx += 1
    result = sum(response_times) / len(response_times)
    return result
    
def parse_date(date_str):
    return mktime_tz(parsedate_tz(date_str))

def unique_contacts(emails):
    unique_contacts = {}
    for email in emails:
        if my_email in email['From']:
            if email['To'] not in unique_contacts:
                unique_contacts[email['To']] = 1
            else:
                unique_contacts[email['To']] += 1
    return unique_contacts

# seperate fist_name, last_name and email address as different columns
def split_name_address(emails):
    name_split_emails = []

    for email_dict in emails:
        if isinstance(email_dict, dict):
            if my_email not in email_dict["From"]:
                first_name, last_name, email_address = split_email_address(email_dict['From'])
                new_item = email_dict.copy()
                new_item['First_Name'] = first_name
                new_item['Last_Name'] = last_name
                new_item['Email'] = email_address
                name_split_emails.append(new_item)
            else:
                first_name, last_name, email_address = split_email_address(email_dict['To'])
                new_item = email_dict.copy()
                new_item['First_Name'] = first_name
                new_item['Last_Name'] = last_name
                new_item['Email'] = email_address
                name_split_emails.append(new_item)
    return name_split_emails


# split full_address into 3 parts
def split_email_address(full_address):
    match = re.match(r'(.+?)\s*<(.+?)>', full_address)
    
    if match:
        name = match.group(1).strip()
        email = match.group(2).strip()
        
         
        name_parts = name.split()
        first_name = name_parts[0]
        last_name = ' '.join(name_parts[1:]) if len(name_parts) > 1 else ''
        
        return first_name, last_name, email
    else:
        return None, None, None

# transform email list such that each unique email address has a single row with multiple email bodies in separate columns
def transform_emails(emails):
    df = pd.DataFrame(emails)
    # drop "subject",'From' ,'To' columns
    df.drop(columns=['Subject', 'From' ,'To'], inplace=True)
    # Group by email address and collect email bodies
    grouped = df.groupby(['First_Name', 'Last_Name', 'Email'])['Body'].apply(list).reset_index()
    grouped['Body'] = grouped['Body'].apply(lambda x: ' '.join(x))
    grouped = grouped.to_dict(orient='records')

    return grouped

# conduct sentiment analysis and add sentiment into last column 'sentiment_analysis'
def groq_sentiment_analysis(emails):
    sentiment_analysis_emails = []

    for email in emails:
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
                "content": email['Body']
            }
        ],
        temperature=1,
        max_tokens=1024,
        top_p=1,
        stream=False,
        stop=None,
        )
        new_item = email.copy()
        new_item['sentiment_analysis'] = chat_completion.choices[0].message.content
        sentiment_analysis_emails.append(new_item)

    
    return sentiment_analysis_emails

# summarize one sentence relationship and add it into last column
def groq_summary_relationship(emails):
    summary_emails = []

    for email in emails:
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
                "content": email['Body']
            }
        ],
        temperature=1,
        max_tokens=1024,
        top_p=1,
        stream=False,
        stop=None,
        )
        new_item = email.copy()
        new_item['summary_relationship'] = chat_completion.choices[0].message.content
        summary_emails.append(new_item)
    
    return summary_emails


def save_to_csv(emails, output_file):
    df = pd.DataFrame(emails)  
    df.to_csv(output_file, index=False)

# Main function
def convert_mbox_to_csv(mbox_file, output_file):
    emails = extract_emails(mbox_file)
    user_response_time = response_time(emails)
    unique_contact_interactions_list = unique_contacts(emails)
    filtered_emails = filter_emails(emails)
    name_split_emails = split_name_address(filtered_emails)
    cleaned_email = transform_emails(name_split_emails)
    sentiment_analysis_emails = groq_sentiment_analysis(cleaned_email)
    summary_emails = groq_summary_relationship(sentiment_analysis_emails)

    save_to_csv(summary_emails, output_file)

# update user's mbox file and email address 
mbox_file = 'dev.mbox'
my_email = 'user@example.com'
output_file = 'output.csv'

client = Groq(
    api_key = os.getenv("GROQ_API_KEY"),
)

convert_mbox_to_csv(mbox_file, output_file)