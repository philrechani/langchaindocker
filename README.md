# Current run instructions:
First make sure that docker desktop is running.

```
cd ./langchaindocker (Wherever you happen to put the repo)
```

then run:
```
docker compose build
docker compose up --detach
```

The app can also be run on your host machine instead of the app container. You still need to run the container so it runs the database.

First, change ```config/CHROMA_CONFIG.py``` so that it uses
```
HOST='localhost'
PORT='49151'
NAME='whatever_you_wish'
```
Then go to  ```app/``` and run```python cli.py```