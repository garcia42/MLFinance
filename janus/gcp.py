import json
import logging
from os import environ

# Check if we're running locally or in GCP
on_localhost = environ.get('K_SERVICE', 'localhost') == 'localhost'

# Set up basic logging handler
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)
logger.propagate = False

# Only import and setup GCP logging if we're not local
if not on_localhost:
    try:
        from google.cloud import logging as gcp_logging, secretmanager_v1 as secretmanager
        handler = gcp_logging.Client().get_default_handler()
        logger.addHandler(handler)
    except Exception as e:
        logger.warning(f"Failed to initialize GCP logging: {e}")

class GcpModule:
    _logging = logger
    _sm = None

    @property
    def logging(self):
        return self._logging

    @property
    def sm(self):
        if not self._sm and not on_localhost:
            try:
                self._sm = secretmanager.SecretManagerServiceClient()
            except Exception as e:
                self.logging.warning(f"Failed to initialize Secret Manager: {e}")
        return self._sm

    @classmethod
    def get_logger(cls):
        return cls._logging

    def get_secret(self, secret_name):
        """
        Fetches secrets from Secret Manager.
        Falls back to environment variables if running locally.

        :param secret_name: name of the secret
        :return: secret value (dict or str)
        """
        if on_localhost:
            # When local, try to get from environment variables
            secret = environ.get(secret_name)
            if not secret:
                self.logging.warning(f"Secret {secret_name} not found in environment")
                return None
        else:
            if not self.sm:
                self.logging.error("Secret Manager not initialized")
                return None
            try:
                secret = self.sm.access_secret_version(name=secret_name).payload.data.decode()
            except Exception as e:
                self.logging.error(f"Failed to fetch secret {secret_name}: {e}")
                return None

        try:
            return json.loads(secret)
        except json.decoder.JSONDecodeError:
            return secret