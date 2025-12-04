import requests
def send_requests(isvc_name: str):
    request_data = {
        'parameters': {'content_type': 'pd'}, 
        'inputs': [{'name': 'season', 'shape': [2], 'datatype': 'UINT64', 'data': [1, 1]}, 
                   {'name': 'holiday', 'shape': [2], 'datatype': 'UINT64', 'data': [0, 0]}, 
                   {'name': 'workingday', 'shape': [2], 'datatype': 'UINT64', 'data': [0, 0]}, 
                   {'name': 'weather', 'shape': [2], 'datatype': 'UINT64', 'data': [1, 1]}, 
                   {'name': 'temp', 'shape': [2], 'datatype': 'FP64', 'data': [9.84, 9.02]}, 
                   {'name': 'atemp', 'shape': [2], 'datatype': 'FP64', 'data': [14.395, 13.635]}, 
                   {'name': 'humidity', 'shape': [2], 'datatype': 'UINT64', 'data': [81, 80]}, 
                   {'name': 'windspeed', 'shape': [2], 'datatype': 'FP64', 'data': [0.0, 0.0]}, 
                   {'name': 'hour', 'shape': [2], 'datatype': 'UINT64', 'data': [0, 1]}, 
                   {'name': 'day', 'shape': [2], 'datatype': 'UINT64', 'data': [1, 1]}, 
                   {'name': 'month', 'shape': [2], 'datatype': 'UINT64', 'data': [1, 1]}]}
   
    headers = {}
    headers["Host"] = f"{isvc_name}.kserve-inference.example.com"
    url = f"http://kserve-gateway.local:30200/v2/models/{isvc_name}/infer"
    result = requests.post(url, json=request_data, headers=headers)
    return result