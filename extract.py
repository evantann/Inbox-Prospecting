import mailbox
import json

SPAM = {"<jobs-listings@linkedin.com>", "<info@email.meetup.com>", "<team@mail.notion.so>", "<no-reply@messages.doordash.com>", "<updates-noreply@linkedin.com>", "<team@hi.wellfound.com>", "<Starbucks@e.starbucks.com>", "<email@washingtonpost.com>", "<messages-noreply@linkedin.com>"}

def clean_headers(msg):
    """
    Remove all headers except To, From, Subject, Date.
    """
    allowed_headers = ['To', 'From', 'Subject', 'Date']
    cleaned_msg = {key: msg.get(key) for key in allowed_headers if msg.get(key)}
    return cleaned_msg

def process_mbox(input_mbox, output_file):
    """
    Process the mbox file to remove unwanted headers and save the result to a JSON file.
    """
    processed_messages = []
    in_mbox = mailbox.mbox(input_mbox)

    for message in in_mbox:
        if message['From'].split()[-1] in SPAM:
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
                    'From': message.get('From'),
                    'To': message.get('To'),
                    'Subject': message.get('Subject'),
                    'Date': message.get('Date'),
                    'Body': combined_payload
                }
                processed_messages.append(email_dict)
            except Exception as e:
                print(f'Error processing message: {e}')
                continue

        else:
            if message.get_content_type() != 'text/plain':
                continue
            # Single-part messages
            email_dict = clean_headers(message)
            email_dict['Body'] = message.get_payload(decode=True).decode('utf-8', errors='replace')
            processed_messages.append(email_dict)
    
    # Save processed messages to a JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(processed_messages, f, ensure_ascii=False, indent=4)

    print(f"Processed mbox file saved to {output_file}")

if __name__ == "__main__":
    input_mbox = 'my.mbox'  # Replace with your input mbox file name
    output_file = 'mbox_extract.json'  # Replace with your output JSON file name
    process_mbox(input_mbox, output_file)