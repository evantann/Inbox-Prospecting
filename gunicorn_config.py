bind = "0.0.0.0:8080"
workers = 8
timeout = 1200
accesslog = 'access.log'
errorlog = 'error.log'
daemon = True

#  run command: gunicorn -c g_config.py app:app