# Inbox_Prospecting

This program is designed to identify and analyze contacts who have a close relationship with the user. It performs sentiment analysis on email content to evaluate each contact's sentiment towards the user and the nature of their relationship.

# Steps:

# Challenges
Problem: Dash runs its own web server.

1. Doesn't natively work with Jinja templates.
2. Had to create individual dash apps for each user.
3. Dash apps had to be created before the Flask server processed any request (no dynamic dashboards).

Iframes resolved 1. Using callbacks solved 2, 3.

# Deploy
https://dev.to/stefanie-a/how-to-deploy-a-flask-app-on-digitalocean-3ib7

# Redis
sudo apt-get update
sudo apt-get install redis-server
sudo service redis-server start
sudo nano /etc/redis/redis.conf
    update bind configuration: bind 0.0.0.0
sudo service redis-server restart