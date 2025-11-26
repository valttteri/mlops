from utils.common_utils import pull_data, preprocess
from mlserver.codecs import PandasCodec
from typing import Dict, Optional
import subprocess
import time
import requests

def prepare_request_data() -> Dict:
    """
    Prepare some data to send in requests to mlserver
    Returns:
        The data to send in the request. It's encoded following the V2 inference protocol
    """
    df = pull_data()
    _, test = preprocess(df)
    test_x = test.drop(["count"], axis=1)

    request_data = test_x.head()

    # Encode the request data following the V2 inference protocol
    encoded_request_data = PandasCodec.encode_request(request_data).dict()
    return encoded_request_data

def run_mlserver(command_to_start_mlserver: str, request_data: Dict) -> Optional[requests.Response]:
    """
    Run mlserver and send a request to the model
    Args:
        command_to_start_mlserver: the command to start mlserver
        request_data: the data to send in the request
    Returns:
        The response from the model (if no error is thrown)
    """
    with subprocess.Popen(command_to_start_mlserver.split(" "), stdout=True) as proc:
        try:
            time.sleep(15)  # Give some time for mlserver to start and load the model
            response = requests.post(
                "http://localhost:8080/v2/models/bike-demand-predictor/infer",
                json=request_data,
            )
            return response
        except Exception as e:
            print(f"ERROR: {e}")
            raise e
        finally:
            proc.terminate()