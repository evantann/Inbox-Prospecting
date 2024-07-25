import icalendar
from datetime import datetime, timedelta
import pandas as pd
import pytz

# Load the ICS file
def load_ics(file_path):
    with open(file_path, 'rb') as f:
        return icalendar.Calendar.from_ical(f.read())

# Extract events from the calendar, and keep meeting time and attendees data
def extract_events(calendar):
    events = []
    for component in calendar.walk():
        if component.name == "VEVENT":
            time = component.get('dtstart').dt
            # summary = str(component.get('summary'))
            attendees = component.get('attendee', [])
            if not isinstance(attendees, list):
                attendees = [attendees]
            attendees = [attendee.to_ical().decode().replace('mailto:', '') for attendee in attendees]
            events.append((time, attendees))
    return events

# Filter meetings within 1 year
def filter_past_year_events(events):
    one_year_ago = datetime.now(pytz.utc) - timedelta(days=365)
    return [event for event in events if event[0].astimezone(pytz.utc) > one_year_ago]  

# Remove group meetings, only keep 1:1 meeting
def remove_group_meetings(events):
    return [event for event in events if len(event[1]) == 2]

# Filter attendees emails, keep only the other person's email, exclude user's 
def keep_other_person_email(events, user_email):
    updated_events = []
    for event in events:
        other_email = [email for email in event[1] if user_email not in email]
        if other_email:
            updated_events.append((event[0], other_email[0]))
    return updated_events

# Remove team members using the same domain as user's
def remove_team_members(events, user_email_domain):
    return [event for event in events if user_email_domain not in event[1]]

# Mapping each contact sentiment by reading csv file from email_sentiment_analysis.py
# def mapping_sentiment(events, sentiment_csv):
#     final_events = []
#     sentiment_df = pd.read_csv(sentiment_csv)
#     for event in events:
#         email = event[1]
#         if email in sentiment_df['email'].values:
#             sentiment = sentiment_df[sentiment_df['email'] == email]['sentiment_analysis'].values[0]
#             if sentiment.lower() != 'negative':
#                 final_events.append((event[0], email, sentiment))
#         else:
#             final_events.append((event[0], email, 'no sentiment'))

#     #Define sorting key
#     def sentiment_sort_key(event):
#         sentiment_priority = {'positive': 1, 'neutral': 2, 'no sentiment': 3}
#         return sentiment_priority.get(event[2].lower(), 4)
    
#     # Sort the events based on the defined key
#     final_events.sort(key=sentiment_sort_key)
#     return final_events

def mapping_sentiment(events, sentiment_csv):
    final_events = []
    sentiment_df = pd.read_csv(sentiment_csv)

    # Create a set of emails that have calendar meetings
    meeting_emails = {event[1] for event in events}
    for index, row in sentiment_df.iterrows():
        # first_name = row['first_name']
        # last_name = row['last_name']
        email = row['email']
        sentiment = row['sentiment_analysis'].lower()
        has_calendar_meeting = email in meeting_emails
        final_events.append((email, sentiment, has_calendar_meeting))

    # Add entries from the events that are not in the sentiment file
    sentiment_emails = set(sentiment_df['email'])
    for event in events:
        email = event[1]
        if email not in sentiment_emails:
            final_events.append((email, 'no email contact no sentiment', True))

    # Define sorting key
    def sentiment_sort_key(event):
        sentiment_priority = {
            ('positive', True): 1,
            ('neutral', True): 2,
            ('positive', False): 3,
            ('no email contact no sentiment', True):4,
            ('neutral', False): 5,
            ('negative', True): 6,
            ('negative', False): 7
        }
        return sentiment_priority.get((event[1], event[2]), 9)
    
    # Sort the events based on the defined key
    final_events.sort(key=sentiment_sort_key)
    return final_events


# Main function
def email_calendar_data_agent(ics_file, user_email, output_csv, sentiment_csv):
    user_email_domain = '@' + user_email.split('@')[1]
    
    calendar = load_ics(ics_file)
    events = extract_events(calendar)
    past_year_events = filter_past_year_events(events)
    one_on_one_meetings = remove_group_meetings(past_year_events)
    one_on_one_meeting_contact = keep_other_person_email(one_on_one_meetings, user_email)
    outsider_meetings = remove_team_members(one_on_one_meeting_contact, user_email_domain)
    final_events = mapping_sentiment(outsider_meetings, sentiment_csv)
    for event in final_events:
        print(event)
    
    # Convert to DataFrame
    df = pd.DataFrame(final_events, columns=['email', 'sentiment', 'has_calendar_meeting'])
    
    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"CSV file saved to {output_csv}")

# update use's files name and email address
ics_file = 'stellahuan2@gmail.com.ics'
user_email = 'stellahuan2@gmail.com'
output_csv = 'email_calendar_data_agent.csv'   
sentiment_csv = "contact_sentiment.csv"

email_calendar_data_agent(ics_file, user_email, output_csv, sentiment_csv)