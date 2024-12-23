FROM python:3.12

COPY requirements.txt .
RUN apt update && \
    apt upgrade --no-install-recommends && \
    apt install --no-install-recommends \
        less \
        && \
    pip install --no-cache-dir -r requirements.txt && \
    rm -rf /var/cache/apt/archives /var/lib/apt/lists/*

# jupyter server -y \
#     --ip 0.0.0.0 \
#     --allow-root \
#     --no-browser \
#     --PasswordIdentityProvider.password_required False \
#     --PasswordIdentityProvider.hashed_password '' \
#     --IdentityProvider.token '' \
#     --ServerApp.allow_origin '*' \
#     --ServerApp.disable_check_xsrf True
WORKDIR /src
CMD ["jupyter", "server", "-y", "--allow-root", "--no-browser", "--ip", "0.0.0.0", "--PasswordIdentityProvider.password_required", "False", "--PasswordIdentityProvider.hashed_password", "", "--IdentityProvider.token", "", "--ServerApp.allow_origin", "*", "--ServerApp.disable_check_xsrf", "True"]
