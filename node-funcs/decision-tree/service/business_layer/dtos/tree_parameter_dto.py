from typing import Optional
from minio import Minio
import pandas as pd
import pickle
from io import BytesIO
from dataclasses import dataclass
from service.business_layer.data_preparation.preparation import prepare_dataset_format
from service.infrastructure_layer.constants import Criterion, Splitter, SplitSize
from service.infrastructure_layer.options.conig import _config

@dataclass(kw_only=True)
class TreeParametersDTO:
    """
    Data class for parameters supporting DecisionTreeClassifier
    """
    test_size: Optional[int] = None
    criterion: Optional[str] = None
    splitter: Optional[str] = None
    max_depth: Optional[int] = None
    min_samples_split: Optional[int] = None
    min_samples_leaf: Optional[int] = None
    min_weight_fraction: Optional[float] = None
    max_features: Optional[str] = None
    max_leaf_nodes: Optional[int] = None
    target: Optional[str] = None
    data: Optional[pd.DataFrame] = None

    @classmethod
    def from_dict(cls, data: dict):
        # Cast all float values to int where the field is expected to be an int
        for key in ['max_depth', 'min_samples_split', 'min_samples_leaf', 'max_leaf_nodes']:
            if key in data and isinstance(data[key], float):
                data[key] = int(data[key])
        return cls(**data)


def read_csv_data_into_df(req):
    client = Minio(
                "minio:24001",
                access_key=req['config']['access_key']['value'],
                secret_key=req['config']['secret_key']['value'],
                secure=False
            )

    bucket_name = req['config']['bucket_name']['value']
    output_file_name = req['outputs']['Dataframe']['destination']

    obj = client.get_object(bucket_name, req['inputs']['Dataframe']['value'])
    pickled_data = obj.read()
    
    #Unpickle the DataFrame
    df = pickle.loads(pickled_data)
    data = prepare_dataset_format(df)

    with BytesIO() as bytes_buffer:
        pickle.dump(df, bytes_buffer)
        bytes_buffer.seek(0)
        client.put_object(bucket_name, output_file_name, bytes_buffer, len(bytes_buffer.getvalue()))
    return data


def create_tree_parameters_dto(req, read_sandbox_data: bool = False):
    """
    Maps the income request into the TreeParametersDTO
    @params: req
    """
    request = req if _config.dev_mode else req
    bucket = request['config']['bucket_name']['value']
    print(f'START create_tree_parameters_dto for {bucket}')

    target_variable = request['inputs']['TargetVariable']['value']
    criterion = 'gini' if request['inputs']['Criterion']['value'] is Criterion.Gini.value else 'entropy'
    splitter = 'best' if request['inputs']['Splitter']['value'] is Splitter.Best.value else 'random'
    testSize = request['inputs']['TestSize']['value'] if request['inputs']['TestSize']['value'] is not None else SplitSize.DEFAULT.value

    print(f'=================> TEST SIZE: {testSize} <====================')
    if not (0 < testSize < 1):
        raise ValueError("Test size must be a float between 0 and 1. For example, 0.2 for 20% test size.")

    test_dataset = 'pvak97nk' if _config.use_pva97kn_dataset else 'breast_cancer_data_a'

    data_dict = {
        "test_size": testSize,
        "criterion": criterion,
        "splitter": splitter,
        "max_depth": request['inputs']['MaxDepth']['value'],
        "min_samples_split": request['inputs']['MinSamplesSplit']['value'],
        "min_samples_leaf": request['inputs']['MinSamplesLeaf']['value'],
        "min_weight_fraction": request['inputs']['MinWeightFraction']['value'],
        "max_features": request['inputs']['MaxFeatures']['value'],
        "max_leaf_nodes": request['inputs']['MaxLeafNodes']['value'],
        "target": target_variable.lower(),
        "data": read_csv_data_into_df(request)
    }
    return TreeParametersDTO.from_dict(data_dict)


def print_parameters(parameters: TreeParametersDTO):
    print()
    print('START Parameter values')
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
    print('END Parameter values')
