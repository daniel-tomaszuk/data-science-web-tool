# data-science-web-tool

### Pre-requirements 
Docker and docker-compose. Please check out Docker documentation for 
instructions how to install it on your local device.

### Start Docker services:
`docker compose up -d db`

### Install requirements
It's preferable to install requirements on dedicated virtualenv.

`pip install -r requirements.txt`

### Create necessary DB migrations
`./manage.py migrate`


### Start server
`./manage.py runserver 8000`
