# Current run instructions:
First make sure that docker desktop is running.

```
cd ./langchaindocker (Wherever you happen to put the repo)
```

then run:
```
docker compose build
docker compose up --detach
````
(I believe docker compose up will build it if it isn't already, so docker compose  build might not be necessary)

The llm-server container should automatically serve ollama

Then you need to find the NAME of the app container. It should be langchaindocker-app-1. Run the following command and the name should be under NAMES

```
docker container ls
```
This will show all running containers

Finally, open a separate terminal and run
```
docker attach langchaindocker-app-1
ollama run mistral
```
(if the NAME is different, use that NAME instead. You can also use another model besides mistral)

When you're done, ctrl-C the server (if it is open) and /bye -> exit the app terminal to leave. The run:
```
docker compose down
```

# Issues/todo
1. It has to install mistral every time. It shouldn't have to do this.
2. Doesn't recognize GPU yet, I'll need to consult here: https://hub.docker.com/r/ollama/ollama to figure out how to do that with multiple GPU types (automatically recognize)
3. Add the scripts to use langchain