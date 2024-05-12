# Use Ubuntu 22.04 LTS as the base image
FROM ubuntu:22.04

ENV USER=chatbot
ENV HOME=/home/$USER

# Install curl and gnupg for downloading and installing Node.js
# Install necessary dependencies for Puppeteer and CUPS
RUN apt-get update && \
    apt-get install -y \
    curl \
    gnupg \
    python3.8 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p $HOME/app

# Set working directory
WORKDIR $HOME/app

# Copy the application code
COPY . .

# Create a non-root user and set permissions
RUN useradd -ms /bin/bash $USER && \
    chown -R $USER:$USER $HOME

# Change to non-root user
USER $USER

# Expose port 8080
EXPOSE 8080

# Install the required Python packages
RUN pip install -r requirements.txt

# Start the flask server
CMD ["python3", "app.py"]
