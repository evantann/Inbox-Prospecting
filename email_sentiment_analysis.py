import mailbox
import re
import csv
import pandas as pd
from collections import defaultdict
import os
from groq import Groq

# check if it's marketing and promotional email using common marketing keyword filters and sender domain
def is_marketing_email(subject, from_address):
    marketing_keywords = ["sale", "discount", "save", "off", "offer", "deal", "clearance",
        "free", "giveaway", "promo", "bogo", "limited time", "hurry",
        "act now", "last chance", "exclusive", "only", "final", "ending soon",
        "while supplies last", "today only", "new", "launch", "event",
        "introducing", "announcement", "coming soon", "just arrived",
        "premiere", "grand opening", "subscribe", "member", "loyalty", "join",
        "insider", "vip", "rewards", "holiday", "black friday", "cyber monday",
        "christmas", "new year", "summer", "winter", "spring", "fall",
        "halloween", "thanksgiving", "valentine’s day", "easter", "buy now",
        "shop", "click", "get", "try", "check out", "download", "rsvp",
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
    
    if any(domain in from_address for domain in marketing_domains):
        return True
    
    return False

# check if an email is automatically sent from system, instead of from a personal one
def is_non_personal_email(subject, from_address):
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
            'subject': message['subject'],
            'from': message['from'],
            'to': message['to'],
            'body': body
        })
 
    return emails

#find contacts user contacted at least 3 times 
def close_contacts(emails):
    sent_counts = defaultdict(int)
    close_contacts_email = []

    # count contaction time for each contact
    for email in emails:
        if email['from'] and my_email in email['from'] and email['to']:
            for recipient in email['to'].split(','):
                sent_counts[recipient.strip()] += 1

    for K, V in sent_counts.items():
        if(V >= 3):
            close_contacts_email.append(K)

    return close_contacts_email

# filter emails to exclude marketing email and non-personal email
# Exclude contacts to whom you have sent less than 3 emails in your entire history. 
def filter_emails(emails):
    filtered_emails = []
    
    for email in emails:
        if is_marketing_email(email['subject'], email['from']):
            continue
        
        if is_non_personal_email(email['subject'], email['from']):
            continue
        
        if email['from']:
            if(email['from'] not in close_contacts(emails)):
                continue
        
        filtered_emails.append(email)
    
    return filtered_emails

# seperate fist_name, last_name and email address as different columns
def split_name_address(emails):
    name_split_emails = []

    for email_dict in emails:
        if isinstance(email_dict, dict): 
            first_name, last_name, email_address = split_email_address(email_dict['from'])
            new_item = email_dict.copy()
            new_item['first_name'] = first_name
            new_item['last_name'] = last_name
            new_item['email'] = email_address
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
    # drop "subject",'from' ,'to' columns
    df.drop(columns=['subject', 'from' ,'to'], inplace=True)

    # Group by email address and collect email bodies
    grouped = df.groupby(['first_name', 'last_name', 'email'])['body'].apply(list).reset_index()

    max_bodies = grouped['body'].apply(len).max()
    new_columns = ['message{}'.format(i+1) for i in range(max_bodies)]
    body_df = pd.DataFrame(grouped['body'].tolist(), columns=new_columns)

    cleaned_df = pd.concat([grouped.drop(columns='body'), body_df], axis=1)
    cleaned_df['combined_message'] = cleaned_df[new_columns].apply(lambda row: ' '.join(row.dropna()), axis=1)
    cleaned_email = cleaned_df.to_dict(orient='records')

    return cleaned_email

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
                "content": email['combined_message']
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
                "content": email['combined_message']
            }
        ],
        temperature=1,
        max_tokens=1024,
        top_p=1,
        stream=False,
        stop=None,
        )
        # print(chat_completion.choices[0].message.content)
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
    filtered_emails = filter_emails(emails)
    name_split_emails = split_name_address(filtered_emails)
    cleaned_email = transform_emails(name_split_emails)
    sentiment_analysis_emails = groq_sentiment_analysis(cleaned_email)
    summary_emails = groq_summary_relationship(sentiment_analysis_emails)

    save_to_csv(summary_emails, output_file)

# update user's mbox file and email address 
mbox_file = 'user_mbox_file_name.mbox'
my_email = 'user_email_address@gmail.com'
output_file = 'output.csv'


client = Groq(
    api_key = "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"
    # api_key = os.environ.get("GROQ_API_KEY"),
)

convert_mbox_to_csv(mbox_file, output_file)




