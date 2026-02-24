import os
import requests
import json
from dotenv import load_dotenv

# 1. Load credentials
load_dotenv()

JIRA_URL = os.getenv("JIRA_URL")
JIRA_EMAIL = os.getenv("JIRA_EMAIL")
JIRA_TOKEN = os.getenv("JIRA_TOKEN")
PROJECT_KEY = os.getenv("JIRA_PROJECT_KEY")

if not all([JIRA_URL, JIRA_EMAIL, JIRA_TOKEN, PROJECT_KEY]):
    print("❌ Error: Missing environment variables. Please check .env file.")
    exit(1)

# Ensure URL has scheme
if not JIRA_URL.startswith("http"):
    JIRA_URL = f"https://{JIRA_URL}"

# 2. Prepare API Request
# NOTE: The endpoint /rest/api/3/search is deprecated (410 Gone).
# Using /rest/api/3/search/jql to succeed.
api_endpoint = f"{JIRA_URL.rstrip('/')}/rest/api/3/search/jql"

headers = {
    "Accept": "application/json",
    "Content-Type": "application/json"
}

payload = {
    "jql": f"project = '{PROJECT_KEY}'",
    "maxResults": 1,
    "fields": ["summary"]
}

print(f"Testing connection to: {api_endpoint}")
print(f"Project: {PROJECT_KEY}")
print("-" * 30)

try:
    response = requests.post(
        api_endpoint,
        json=payload,
        headers=headers,
        auth=(JIRA_EMAIL, JIRA_TOKEN)
    )

    if response.status_code == 200:
        data = response.json()
        issues = data.get("issues", [])
        if issues:
            issue = issues[0]
            key = issue.get("key")
            summary = issue.get("fields", {}).get("summary")
            print(f"✅ Success! Connected to Jira.")
            print(f"Issue found: {key} - {summary}")
        else:
            print(f"✅ Success! Connected to Jira, but no issues found in project '{PROJECT_KEY}'.")
    else:
        print(f"❌ Connection Failed using REST API v3.")
        print(f"Status Code: {response.status_code}")
        try:
            print(f"Error Message: {json.dumps(response.json(), indent=2)}")
        except:
            print(f"Error Text: {response.text}")
            
except Exception as e:
    print(f"❌ unexpected Error: {str(e)}")
