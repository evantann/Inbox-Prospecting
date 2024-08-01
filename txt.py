import mailbox

SPAM = {
    "<jobs-listings@linkedin.com>", "<info@email.meetup.com>", "<team@mail.notion.so>",
    "<no-reply@messages.doordash.com>", "<updates-noreply@linkedin.com>", "<team@hi.wellfound.com>",
    "<Starbucks@e.starbucks.com>", "<email@washingtonpost.com>", "<messages-noreply@linkedin.com>"
}

def process_mbox(input_mbox):
    """
    Process the mbox file to remove unwanted headers and save the result to a text file.
    """
    in_mbox = mailbox.mbox(input_mbox)
    
    with open('txt/email.txt', 'w', encoding='utf-8') as f:
        for message in in_mbox:
            if message['From'].split()[-1] in SPAM:
                continue

            if message.is_multipart():
                for part in message.walk():
                    if part.get_content_type() == 'text/plain':
                        # Write headers
                        f.write("Headers:\n")
                        for header in part.keys():
                            f.write(f"{header}: {part[header]}\n")
                        
                        # Write payload
                        f.write("\nPayload:\n")
                        f.write(part.get_payload(decode=True).decode('utf-8', errors='ignore') + '\n')
                        
                        # Separate different parts of the same email
                        f.write("\n" + "="*50 + "\n")

    print("Processed mbox file saved to email.txt")

if __name__ == "__main__":
    input_mbox = 'my.mbox'  # Replace with your input mbox file name
    process_mbox(input_mbox)
