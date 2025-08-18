import itertools
import time

import os

# Read environment variable
api_key1 = os.getenv("GENAI_API_KEY_1")
api_key2 = os.getenv("GENAI_API_KEY_2")
api_key3 = os.getenv("GENAI_API_KEY_3")

API_KEYS  = []

for api_key in [api_key1, api_key2, api_key3]:
    if api_key:
        print("API Key found:", api_key)
        API_KEYS.append({"key": api_key, "req_timestamps": []})        
    else:
        print("API Key not set.")
        
key_cycle = itertools.cycle(API_KEYS)
MAX_REQS_PER_MIN = 5


def cleanup_usage(key_info):
    """Remove requests older than 60s."""
    now = time.time()
    key_info["req_timestamps"] = [
        t for t in key_info["req_timestamps"] if now - t < 60
    ]


def get_api_key(auto_wait=True):
    """Return an API key that has quota. Waits if needed."""
    while True:
        for _ in range(len(API_KEYS)):
            key_info = next(key_cycle)
            cleanup_usage(key_info)

            if len(key_info["req_timestamps"]) < MAX_REQS_PER_MIN:
                key_info["req_timestamps"].append(time.time())
                return key_info["key"]

        if auto_wait:
            # Find soonest timestamp that will expire
            next_free_time = min(
                min(k["req_timestamps"]) for k in API_KEYS if k["req_timestamps"]
            )
            sleep_for = max(0, 60 - (time.time() - next_free_time))
            print(f"⏳ All keys exhausted. Waiting {sleep_for:.1f}s...")
            time.sleep(sleep_for + 0.1)
        else:
            raise RuntimeError("🚨 All API keys hit 5 req/min limit")
