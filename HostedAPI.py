import requests
import sys
import json

API_URL = "https://on-demand.gravity-ai.com/"
API_CREATE_JOB_URL = API_URL + 'api/v1/jobs'
API_GET_JOB_RESULT_URL = API_URL + 'api/v1/jobs/result-link'
API_KEY = "GAI-wmBiBQ7e.bm3OFeWHioeHUprWJ30ij218sF-GAc"  # Use API Keys tab to generate

config = {
    "version": "0.0.1",
    # Optional - if omitted, the latest version will be used; Set to a specific version number (i.e. 0.0.1, 0.0.2, 1.0.1, etc. Check versions on Hosted Inference page Version dropdown)
    "mimeType": "application/json; header=present",  # 'text/plain', etc. Change based on your file type
}

requestHeaders = {
    'x-api-key': API_KEY
}


def postJob(inputFilePath):
    # Post a new job (file) to the api
    inputFile = open(inputFilePath, 'rb')
    files = {
        "file": inputFile,
    }

    data = {
        'data': json.dumps(config)
    }

    print("Creating job...")
    r = requests.request("POST", API_CREATE_JOB_URL, headers=requestHeaders, data=data, files=files)
    print(r.status_code)
    result = r.json()
    if (result.get('isError', False)):
        print("Error: " + result.get('errorMessage'))
    #     raise Exception("Error: " + result.errorMessage)
    if (result.get('data').get('statusMessage') != "success"):
        print("Job Failed: " + result.get('data').get('errorMessage'))
        raise Exception("Job Failed: " + result.get('data').get('errorMessage'))
    return result.get('data').get('id')


def downloadResult(jobId, outFilePath):
    url = API_GET_JOB_RESULT_URL + "/" + jobId
    r = requests.request("GET", url, headers=requestHeaders)
    link = r.json()
    if (link.get('isError')):
        print("Error: " + link.get('errorMessage'))
        raise Exception("Error: " + link.get('errorMessage'))

    result = requests.request("GET", link.get('data'))
    open(outFilePath, 'wb').write(result.content)


# jobId = postJob(sys.argv[1])
# downloadResult(jobId, sys.argv[2])