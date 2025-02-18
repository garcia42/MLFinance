FROM ghcr.io/gnzsnz/ib-gateway:latest

# Install Python and required packages
USER root
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-full \
    && rm -rf /var/lib/apt/lists/*

# Add your Python application
COPY . /home/ibgateway/app
WORKDIR /home/ibgateway/app

# Create and activate virtual environment
RUN python3 -m venv /home/ibgateway/app/venv
ENV PATH="/home/ibgateway/app/venv/bin:$PATH"

# Install Python dependencies in the virtual environment
RUN pip install --no-cache-dir setuptools && \
    pip install --no-cache-dir -r requirements.txt

# Create a startup script that runs both IB Gateway and your app
RUN echo '#!/bin/bash\n\
/home/ibgateway/scripts/run.sh & \n\
sleep 30 && echo "Starting Python app..." && \n\
/home/ibgateway/app/venv/bin/python /home/ibgateway/app/app.py\n\
' > /home/ibgateway/app/start.sh && \
chmod +x /home/ibgateway/app/start.sh

# Fix permissions
RUN chown -R ibgateway:ibgateway /home/ibgateway/app

# Switch back to the ibgateway user
USER ibgateway

# Use the startup script as the entrypoint
ENTRYPOINT ["/home/ibgateway/app/start.sh"]