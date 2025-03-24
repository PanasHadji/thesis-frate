import pickle

from minio import Minio
from minio.error import S3Error
from io import BytesIO

import pandas as pd

from service.business_layer.modelling.business import DecisionTreeService
from service.business_layer.dtos.tree_parameter_dto import create_tree_parameters_dto



def handle_decision_tree_process(req):
    """
    Decision Tree
    Impute missing values in a DataFrame read from MinIO and upload the imputed DataFrame back to MinIO.
    """
    try:
        # Extract configuration from parsed JSON input
        access_key = req.json['config']['access_key']['value']
        secret_key = req.json['config']['secret_key']['value']
        bucket_name = req.json['config']['bucket_name']['value']

        dt_service = DecisionTreeService()

        # Create a MinIO client
        client = Minio(
            "minio:24001",
            access_key=access_key,
            secret_key=secret_key,
            secure=False
        )

        print('START Print object')
        print(req.json['inputs'])

        # Extract input and output file names directly without parsing
        parameters = create_tree_parameters_dto(req)

        print()
        print('***********')
        print('START Ready to print parameter values...')
        print(f"Criterion: {parameters.criterion}")
        print(f"Splitter: {parameters.splitter}")
        print(f"Max Depth: {parameters.max_depth}")
        print(f"Min Samples Split: {parameters.min_samples_split}")
        print(f"Min Samples Leaf: {parameters.min_samples_leaf}")
        print(f"Min Weight Fraction: {parameters.min_weight_fraction}")
        print(f"Max Features: {parameters.max_features}")
        print(f"Max Leaf Nodes: {parameters.max_leaf_nodes}")
        print(f"Target: {parameters.target}")
        print(f"Previous Node Input: {parameters.data}")
        print()
        print('************')
        print('END')

        print()
        print()
        print('START Preparing DataFrame')
        print()
        print()
        df = pd.DataFrame('data.csv')
        print(df)
        # table_str = tabulate(df, headers='keys', tablefmt='grid')
        # print(table_str)


        input_file_name = 'data'
        output_file_name = req.json['outputs']['Dataframe']['destination']

        # Read data from pickled file in MinIO
        obj = client.get_object(bucket_name, input_file_name)
        pickled_data = obj.read()
        print()
        print()
        print('Pickled Data:')
        print(pickled_data)

        # return True

        # Unpickle the DataFrame
        df = pickle.loads(pickled_data)

        # Pickle the modified DataFrame
        with BytesIO() as bytes_buffer:
            pickle.dump(df, bytes_buffer)
            bytes_buffer.seek(0)
            client.put_object(bucket_name, output_file_name, bytes_buffer, len(bytes_buffer.getvalue()))

        # Return the output file path
        return output_file_name
    except KeyError as exc:
        raise RuntimeError(f"Missing key in JSON input: {exc}")
    except S3Error as exc:
        raise RuntimeError(f"An error occurred while reading from or uploading to MinIO: {exc}")
    except Exception as exc:
        raise RuntimeError(f"Error processing file: {exc}")
