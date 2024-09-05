# Inbox Prospecting

![alt text][dashboard]

[dashboard]: https://github.com/evantann/Inbox-Prospecting/blob/master/Screenshot%202024-08-30%20233448.png

This web app helps users identify contacts with whom they have a close relationship with by analyzing their email inbox (mbox file). Users upload their mbox file for processing after which, an interactive dashboard will display the results. Metrics such as average response time, sentiment, and interaction frequency are factors in evaluating a contact's relationship with the user. Users can upload multiple inboxes and access a personalized dashboard for each account. 

# Tech Stack:
Front End: Plotly Dash  
Back End: Flask  
Database: Redis, PostgreSQL (Supabase)  
Deployment: Docker, Nginx/Gunicorn, Digital Ocean  
NLP: spacy, textblob, Hugging Face (https://huggingface.co/mrm8488/bert-tiny-finetuned-sms-spam-detection)

# Server Environment
### Nginx
1. sudo apt install nginx -y
2. sudo systemctl start nginx
3. sudo nginx -s reload

### Redis
1. sudo apt-get update
2. sudo apt-get install redis-server
3. sudo service redis-server start
4. sudo nano /etc/redis/redis.conf
    `bind 127.0.0.1 ::0`
5. sudo service redis-server restart  
   
### SSL Certificate
1. sudo apt-get update
2. sudo apt-get install certbot python3-certbot-nginx
3. sudo certbot --nginx -d `your_ip_address`
