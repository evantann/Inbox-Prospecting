bind = "0.0.0.0:8080"
workers = 8
timeout = 1200
accesslog = 'access.log'
errorlog = 'error.log'
daemon = True

# run command: gunicorn -c gunicorn_config.py app:app
# cli commmand: gunicorn -b 0.0.0.0:8080 -w 8 --timeout 1200 app:app