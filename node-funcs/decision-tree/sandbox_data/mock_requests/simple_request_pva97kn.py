

def mock_simple_request_pva97kn():
    json_string = {
        "inputs": {
            "TestSize": {
                "type": "literalJSON",
                "value": 0.2
            },
            "Criterion": {
                "type": "literalJSON",
                "value": 1
            },
            "Splitter": {
                "type": "literalJSON",
                "value": 1
            },
            "MaxDepth": {
                "type": "literalJSON",
                "value": 6
            },
            "MinSamplesSplit": {
                "type": "literalJSON",
                "value": 2.0
            },
            "MinSamplesLeaf": {
                "type": "literalJSON",
                "value": 5
            },
            "MinWeightFraction": {
                "type": "literalJSON",
                "value": 3.0
            },
            "MaxFeatures": {
                "type": "literalJSON",
                "value": "log2"
            },
            "MaxLeafNodes": {
                "type": "literalJSON",
                "value": 10
            },
            "TargetVariable": {
                "type": "literalJSON",
                "value": "TARGET_B"
            },
            "PreviousInput": {
                "type": "literalJSON",
                "value": 'null'
            }
        },
        "config": {
            "access_key": {
                "type": "literalJSON",
                "value": "minio_admin"
            },
            "secret_key": {
                "type": "literalJSON",
                "value": "minio_admin_password"
            },
            "bucket_name": {
                "type": "literalJSON",
                "value": "stef-workflow-artifacts"
            }
        },
        "outputs": {
            "Dataframe": {
                "type": "pickleDf",
                "destination": "95ecfdcbd3949c7c-204acf7226525405/output-Dataframe"
            }
        }
    }
    return json_string