FROM ghcr.io/gnzsnz/ib-gateway:latest

# Install Python and required packages
USER root
RUN apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y \
    python3.9 \
    python3.9-venv \
    python3.9-dev \
    net-tools \
    procps \
    && rm -rf /var/lib/apt/lists/*

# Add your Python application
COPY . /home/ibgateway/app
WORKDIR /home/ibgateway/app

# Create and activate virtual environment with Python 3.9
RUN python3.9 -m venv /home/ibgateway/app/venv
ENV PATH="/home/ibgateway/app/venv/bin:$PATH"

# Install Python dependencies in the virtual environment
RUN pip install --no-cache-dir setuptools && \
    pip install --no-cache-dir -r requirements.txt

# Create an improved startup script that runs both IB Gateway and your app
RUN echo '#!/bin/bash\n\
# Start the IB Gateway in the background\n\
/home/ibgateway/scripts/run.sh & \n\
IB_GATEWAY_PID=$!\n\
\n\
# Wait for IB Gateway to initialize (up to 5 minutes)\n\
echo "Waiting for IB Gateway to initialize..."\n\
MAX_ATTEMPTS=10\n\
ATTEMPT=0\n\
while [ $ATTEMPT -lt $MAX_ATTEMPTS ]; do\n\
    # Check if port 4002 is in LISTEN state\n\
    if grep -q "00000000:0FA4" /proc/net/tcp; then\n\
        echo "IB Gateway is ready and listening on port 4002!"\n\
        break\n\
    fi\n\
    \n\
    # Check IB Gateway process is still running\n\
    if ! ps -p $IB_GATEWAY_PID > /dev/null; then\n\
        echo "ERROR: IB Gateway process died unexpectedly"\n\
        if [ -f /home/ibgateway/Jts/launcher.log ]; then\n\
            echo "Last 20 lines of the IB Gateway logs:"\n\
            tail -n 20 /home/ibgateway/Jts/launcher.log\n\
        fi\n\
        exit 1\n\
    fi\n\
    \n\
    ATTEMPT=$((ATTEMPT+1))\n\
    echo "Attempt $ATTEMPT/$MAX_ATTEMPTS: IB Gateway not ready yet, waiting 10 seconds..."\n\
    sleep 10\n\
done\n\
\n\
if [ $ATTEMPT -eq $MAX_ATTEMPTS ]; then\n\
    echo "WARNING: IB Gateway initialization timeout, but continuing anyway..."\n\
    echo "Last 20 lines of the IB Gateway logs:"\n\
    tail -n 20 /home/ibgateway/Jts/launcher.log\n\
fi\n\
\n\
echo "Starting Python app..."\n\
/home/ibgateway/app/venv/bin/python /home/ibgateway/app/app.py\n\
' > /home/ibgateway/app/start.sh && \
chmod +x /home/ibgateway/app/start.sh

# Fix permissions
RUN chown -R ibgateway:ibgateway /home/ibgateway/app

# Switch back to the ibgateway user
USER ibgateway

# Use the startup script as the entrypoint
ENTRYPOINT ["/home/ibgateway/app/start.sh"]
