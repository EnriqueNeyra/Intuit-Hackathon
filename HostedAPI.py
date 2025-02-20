import requests
import sys
import json

API_URL = "https://on-demand.gravity-ai.com/"
API_CREATE_JOB_URL = API_URL + 'api/v1/jobs'
API_GET_JOB_RESULT_URL = API_URL + 'api/v1/jobs/result-link'
API_KEY = "GAI-wmBiBQ7e.bm3OFeWHioeHUprWJ30ij218sF-GAc"  # Use API Keys tab to generate

# Update configuration if necessary per documentation:
config = {
    "version": "0.0.1",
    "mimeType": "application/json",  # Remove 'header=present' if not required
}

requestHeaders = {
    'x-api-key': API_KEY
}

def postJob(inputFilePath):
    # Use a context manager to ensure the file is closed
    with open(inputFilePath, 'rb') as inputFile:
        files = {
            "file": inputFile,
        }
        # Note: Many docs require the configuration JSON to be sent under 'config'
        data = {
            'config': json.dumps(config)
        }

        print("Creating job...")
        r = requests.post(API_CREATE_JOB_URL, headers=requestHeaders, data=data, files=files)
        print("Status code:", r.status_code)
        try:
            result = r.json()
        except Exception as e:
            raise Exception("Failed to parse JSON response: " + str(e))
        
        if result.get('isError', False):
            error_message = result.get('errorMessage', 'Unknown error')
            print("Error: " + error_message)
            raise Exception("Error: " + error_message)
        
        # Check that statusMessage exists and equals "success"
        data_field = result.get('data')
        if not data_field or data_field.get('statusMessage') != "success":
            error_message = data_field.get('errorMessage', 'Job failed without a specific error message') if data_field else 'No data in response'
            print("Job Failed: " + error_message)
            raise Exception("Job Failed: " + error_message)
        return data_field.get('id')

def downloadResult(jobId):
    # Construct the URL to fetch the result link
    url = API_GET_JOB_RESULT_URL + "/" + jobId
    r = requests.get(url, headers=requestHeaders)
    link_response = r.json()
    
    if link_response.get('isError', False):
        error_message = link_response.get('errorMessage', 'Unknown error')
        print("Error: " + error_message)
        raise Exception("Error: " + error_message)
    
    # Retrieve the actual result from the URL provided in the 'data' field
    result_response = requests.get(link_response.get('data'), headers=requestHeaders)
    
    try:
        result_json = result_response.json()
    except Exception as e:
        raise Exception("Failed to parse result JSON: " + str(e))
    
    # Extract and return the 'reply_message' from the result JSON
    if 'reply_message' in result_json:
        return result_json['reply_message']
    else:
        raise Exception("No 'reply_message' found in result JSON")

# Uncomment and pass command-line arguments if running from terminal:
# jobId = postJob(sys.argv[1])
# downloadResult(jobId, sys.argv[2])
