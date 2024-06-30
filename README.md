# Inbox_Prospecting

This program is designed to identify and analyze contacts who have a close relationship with the user. By leveraging Groq AI LLM, it performs sentiment analysis on email content to evaluate each contact's sentiment towards the user and the nature of their relationship. Furthermore, it utilizes Google Calendar data to identify contacts who have had one-on-one meetings with the user and maintain a non-negative sentiment towards them. The program comprises two Python scripts: email_sentiment_analysis.py, which conducts sentiment analysis on email content, and email_calendar_data_agent.py, which assigns sentiment tags to participants based on Google Calendar data.

# email_sentiment_analysis.py
Input: user's email content .mbox file, user's email address
output: csv file with contact first name, contact last name, contact email address, each email message, sentiment, relationship summary

# email_calendar_data_agent.py
Input: user's google calendar .ics file, user's email address, email_sentiment_analysis.csv file
output: email_calendar_data_agent.csv file with contact email address, sentiment, and has_calendar_meeting

# Steps:
1. Install the Groq and icalendar Python library before starting: 
pip install groq
pip install icalendar
2. download the user's email content .mbox file and provide the user's email address in email_sentiment_analysis.py
3. run email_sentiment_analysis.py to get email_sentiment_analysis.csv file
4. download the user's google calendar .ics file and provide the user's email address and email_sentiment_analysis.csv file name in email_calendar_data_agent.py
5. run email_calendar_data_agent.py to get one_on_one meetings.csv



