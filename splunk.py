import pandas as pd
import requests
import json

# Splunk Configuration
splunk_url = "https://splunk-server:8088/services/collector"
splunk_token = "your_hec_token"
headers = {
    "Authorization": f"Splunk {splunk_token}",
    "Content-Type": "application/json"
}

# Read Excel File
excel_file = "report.xlsx"
df = pd.read_excel(excel_file)

# Convert to JSON format for Splunk
for _, row in df.iterrows():
    event_data = row.to_dict()  # Convert each row to dictionary
    data = {"event": event_data, "sourcetype": "excel_reports"}
    response = requests.post(splunk_url, headers=headers, data=json.dumps(data))

    print(response.text)  # Print response from Splunk
