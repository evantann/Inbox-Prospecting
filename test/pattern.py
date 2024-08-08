import re
import json
import mailbox

input_mbox = '../school.mbox'

def clean_headers(msg):
    try:
        allowed_headers = ["To", "From", "Subject", "Date"]
        cleaned_msg = {key: str(msg.get(key)) for key in allowed_headers if msg.get(key)}
        return cleaned_msg
    except Exception as e:
        print(f'Error cleaning headers: {e}')

def extract_mbox(input_mbox):
    try:
        processed_messages = []
        in_mbox = mailbox.mbox(input_mbox)
        count = 0
        on_wrote_pattern = r'^(.*?)(On .*? wrote:)'
        from_pattern = r'^(.*?)(From: [^<]+<[^>]+>)'
        forwarded_message_pattern = r'^-{10,} Forwarded message -{10,}$'
        
        with open('pattern.txt', 'w') as f:
            for message in in_mbox:
                print(f"Extracting message #{count}")
                count += 1
                
                if message.is_multipart():
                    combined_payload = []
                    try:
                        for part in message.walk():
                            if part.get_content_type() == 'text/plain':
                                payload = str(part.get_payload(decode=True).decode('utf-8', errors='replace'))
                                if payload:
                                    payload = payload.replace('\r\n', ' ')
                                    if re.match(forwarded_message_pattern, payload, re.DOTALL):
                                        payload = payload
                                        f.write(payload + '\n')
                                        f.write('\n' + '+' * 75 + '--MESSAGE--' + '+' * 75 + '\n')
                                    elif re.match(on_wrote_pattern, payload, re.DOTALL):
                                        match = re.match(on_wrote_pattern, payload, re.DOTALL)
                                        if match:
                                            payload = match.group(1)
                                            f.write(payload + '\n')
                                            f.write('\n' + '+' * 75 + '--MESSAGE--' + '+' * 75 + '\n')
                                    
                                    elif re.match(from_pattern, payload, re.DOTALL):
                                        match = re.match(from_pattern, payload, re.DOTALL)
                                        if match:
                                            payload = match.group(1)
                                            f.write(payload + '\n')
                                            f.write('\n' + '+' * 75 + '--MESSAGE--' + '+' * 75 + '\n')
                                    combined_payload.append(payload)
                        
                        combined_payload = '\n'.join(combined_payload)
                        
                        email_dict = {
                            "From": str(message.get('From')),
                            "To": str(message.get('To')),
                            "Subject": str(message.get('Subject')),
                            "Date": str(message.get('Date')),
                            "Body": combined_payload
                        }

                        processed_messages.append(email_dict)

                    except Exception as e:
                        print(f'Error processing multipart message: {e}')
                        continue

                else:
                    if message.get_content_type() != 'text/plain':
                        continue

                    email_dict = clean_headers(message)
                    payload = str(message.get_payload(decode=True).decode('utf-8', errors='replace'))
                    if not payload:
                        continue

                    payload = payload.replace('\r\n', '')

                    if re.match(forwarded_message_pattern, payload, re.DOTALL):
                        payload = payload
                        f.write(payload + '\n')
                        f.write('\n' + '+' * 75 + '--MESSAGE--' + '+' * 75 + '\n')

                    elif re.match(on_wrote_pattern, payload, re.DOTALL):
                        match = re.match(on_wrote_pattern, payload, re.DOTALL)
                        if match:
                            payload = match.group(1)
                            f.write(payload + '\n')
                            f.write('\n' + '+' * 75 + '--MESSAGE--' + '+' * 75 + '\n')
                    
                    elif re.match(from_pattern, payload, re.DOTALL):
                        match = re.match(from_pattern, payload, re.DOTALL)
                        if match:
                            payload = match.group(1)
                            f.write(payload + '\n')
                            f.write('\n' + '+' * 75 + '--MESSAGE--' + '+' * 75 + '\n')

                    email_dict["Body"] = payload
                    processed_messages.append(email_dict)

        with open('pattern_extract.json', 'w') as f:
            json.dump(processed_messages, f, indent=4)
        
    except Exception as e:
        print(f'Error extracting mbox: {e}')

extract_mbox(input_mbox)