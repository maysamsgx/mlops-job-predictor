"""
Smoke tests for the API.
Validates basic connectivity and response structure.
"""
import sys
import subprocess
import time
import requests

def run_smoke_test():
    """
    Checks if the service is up and responding.
    """
    print("Waiting for service to start...")
    time.sleep(5)

    url = "http://localhost:8000/predict"
    payload = {
        "skills": "Python,Docker",
        "qualification": "B.Tech",
        "experience_level": "Junior"
    }

    try:
        response = requests.post(url, json=payload, timeout=5)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")

        if response.status_code == 200:
            print("[OK] Smoke Test Passed")
            sys.exit(0)
        else:
            print("[FAIL] Smoke Test Failed")
            sys.exit(1)

    except Exception as e:
        print(f"[FAIL] Connection Failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_smoke_test()
