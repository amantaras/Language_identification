"""
test_simple_connection.py

Test minimal HTTP connection to STT endpoints.
"""

import requests
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format="[%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def test_http_connection(host):
    """Test a minimal HTTP connection to verify the container is responsive."""
    # Convert WebSocket URL to HTTP
    if host.startswith("ws://"):
        host = f"http://{host[5:]}"
    elif host.startswith("wss://"):
        host = f"https://{host[6:]}"

    # Try a simple HEAD request to check server availability
    log.info(f"Testing HTTP connection to {host}")

    try:
        # First try without any path
        response = requests.head(host, timeout=3)
        log.info(f"Status code for base URL: {response.status_code}")
        if response.status_code < 400:  # Any non-error response
            log.info("Success! Container is responding to HTTP requests.")
            return True
    except requests.exceptions.RequestException as e:
        log.warning(f"Base URL request failed: {e}")

    # Try with a typical health endpoint
    try:
        health_url = f"{host}/healthz"
        response = requests.head(health_url, timeout=3)
        log.info(f"Status code for health endpoint: {response.status_code}")
        if response.status_code < 400:
            log.info("Success! Container health endpoint is responding.")
            return True
    except requests.exceptions.RequestException as e:
        log.warning(f"Health endpoint request failed: {e}")

    log.error("Failed to connect to container via HTTP")
    return False


if __name__ == "__main__":
    log.info("=== TESTING SIMPLE HTTP CONNECTION TO STT CONTAINERS ===")

    # Test English container
    english_host = "ws://localhost:5004"
    log.info(f"\nTesting English container: {english_host}")
    english_success = test_http_connection(english_host)

    # Test Arabic container
    arabic_host = "ws://localhost:5005"
    log.info(f"\nTesting Arabic container: {arabic_host}")
    arabic_success = test_http_connection(arabic_host)

    # Summarize results
    if english_success and arabic_success:
        log.info("\n✅ Both containers are responding to HTTP requests")
    elif english_success:
        log.info("\n⚠️ Only the English container is responding")
    elif arabic_success:
        log.info("\n⚠️ Only the Arabic container is responding")
    else:
        log.error("\n❌ Neither container is responding to HTTP requests")
