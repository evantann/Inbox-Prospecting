import mailbox

input_mbox = 'school.mbox'

SPAM = {"jobs-listings@linkedin.com", "info@email.meetup.com", "team@mail.notion.so", "no-reply@messages.doordash.com", "updates-noreply@linkedin.com", "team@hiwellfound.com", "Starbucks@e.starbucks.com", "email@washingtonpost.com", "messages-noreply@linkedin.com", "rewards@e.starbucks.com", "info@meetup.com", "college@coll.herffjones.com", "venmo@email.venmo.com", "aws-marketing-email-replies@amazon.com", "chen.li@rexpandjob.com", "invitations@linkedin.com", "members@respond.kp.org", "no-reply@doordash.com", "bankofamerica@emcom.bankofamerica.com", "learn@itr.mail.codecademy.com", "noreply@alliance-mail.oa-bsa.org", "solatwestvillage@emailrelay.com", "alexanderqluong@gmail.com", "no-reply@modernmsg.com"}

def clean_headers(msg):
    try:
        allowed_headers = ['To', 'From', 'Subject', 'Date']
        cleaned_msg = {key: str(msg.get(key)) for key in allowed_headers if msg.get(key)}
        return cleaned_msg
    except Exception as e:
        print(f'Error cleaning headers: {e}')

def extract_mbox(input_mbox):
    try:
        processed_messages = []
        in_mbox = mailbox.mbox(input_mbox)
        count = 0
        
        with open('iter.txt', 'w') as f:
            for message in in_mbox:
                print(f"Extracting message #{count}")
                count += 1
                from_address = str(message.get('From', ''))
                if from_address:
                    if any(spam in from_address for spam in SPAM):
                        continue
                
                if message.is_multipart():
                    combined_payload = []
                    try:
                        for part in message.walk():
                            if part.get_content_type() == 'text/plain':
                                payload = part.get_payload(decode=True).decode('utf-8', errors='replace')
                                combined_payload.append(str(payload))
                                f.write(str(payload))
                                f.write('\n' + 'x' * 75 + '--PART--' + 'x' * 75 + '\n')
                        
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
                    email_dict['Body'] = str(message.get_payload(decode=True).decode('utf-8', errors='replace'))
                    f.write(email_dict['Body'])
                    processed_messages.append(email_dict)
                
                f.write('\n' + '+' * 75 + '--MESSAGE--' + '+' * 75 + '\n')

        return processed_messages
    
    except Exception as e:
        print(f'Error extracting mbox: {e}')

extract_mbox(input_mbox)