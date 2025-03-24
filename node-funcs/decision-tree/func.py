# from service.decision_tree_service import decision_tree
import json

from sandbox_data.mock_requests.simple_request import mock_simple_request
from sandbox_data.mock_requests.simple_request_pva97kn import mock_simple_request_pva97kn
from service.business_layer.pipline import execute_pipeline
from service.infrastructure_layer.options.minio_options import MinIoConfig
from service.infrastructure_layer.options.conig import _config

# if __name__ == '__main__':
#     res = mock_simple_request()
#     #res = mock_simple_request_pva97kn()
#     execute_pipeline(res)

# Main function to handle incoming requests.
def main(context):
    """
    Main function to handle incoming requests.
    """

    if 'request' in context.keys():
        if context.request.method == "POST":
            print('==> START Decision Tree <==')
            print('START Print context')
            print(context.request)
            print()
            print()

            request = context.request.json
            result = execute_pipeline(request)
            print('==> END Decision Tree <==')
            return context.request.method, 200
        elif context.request.method == "GET":
            return context.request.method, 200
    else:
        print("Empty request", flush=True)
        return "{}", 200
