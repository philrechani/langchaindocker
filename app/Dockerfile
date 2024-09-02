# Use the appropriate base image
FROM ubuntu:latest

# Install necessary packages
RUN apt-get update && apt-get install -y curl iputils-ping python3 python3-pip python3-venv

# Install NVM
RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.3/install.sh | bash

# Set environment variables for NVM, Node.js, and npm
ENV NVM_DIR="/root/.nvm"
ENV NODE_VERSION="lts/*"
ENV PATH="$NVM_DIR/versions/node/$NODE_VERSION/bin:$PATH"

# Install Node.js and npm using NVM
RUN bash -c "source $NVM_DIR/nvm.sh && nvm install --lts && nvm alias default node"

WORKDIR /usr/src/app

# Copy package files and install dependencies
COPY package*.json ./
RUN bash -c "source $NVM_DIR/nvm.sh && npm install"

# Copy the rest of your application code
COPY . .

# Set up Python virtual environment and install dependencies
RUN python3 -m venv venv
RUN bash -c ". venv/bin/activate && pip install --upgrade pip && pip install -r requirements.txt"

# Install additional software
RUN curl -fsSL https://ollama.ai/install.sh | sh

# Specify the command to run your application
