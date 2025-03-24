from service.business_layer.data_preparation.preparation import prepare_data
from service.business_layer.data_preparation.preparation_helper import map_parameters_into_dictionary
from service.business_layer.dtos.model_data_dto import Dataset
from service.business_layer.dtos.tree_parameter_dto import create_tree_parameters_dto, print_parameters
from service.business_layer.evaluation.custom_tree_interpreter import TreeInterpreter
from service.business_layer.evaluation.imbalance_detector import ImbalanceDetector
from service.business_layer.modelling.decision_tree_classifier import decision_tree_classifier
from service.infrastructure_layer.external.minio import MinIoClient
from service.infrastructure_layer.file_manager import TempFileManager
from service.infrastructure_layer.pdf_reporter import PdfBuilder
from service.infrastructure_layer.options.conig import _config

def execute_pipeline(request):

    output_file_name = get_bucket_folder_from_request(request['outputs']['Dataframe']['destination'])
    print()
    print()
    print()
    print('**************************************')
    print('**************************************')
    print('**************************************')
    print(f'OUTPUT FILE NAME {output_file_name}')
    print()
    print('**************************************')
    print('**************************************')
    print('**************************************')

    print('stef-decision-tree.execute_pipeline')
    # Extract configuration from parsed JSON input.
    access_key = request['config']['access_key']['value']
    print(f'access_key: {access_key}')
    secret_key = request['config']['secret_key']['value']
    print(f'secret_key: {secret_key}')
    bucket_name = request['config']['bucket_name']['value']
    print(f'bucket_name: {bucket_name}')
    minio_client = MinIoClient(access_key, secret_key, bucket_name, output_file_name)

    pdf_report_builder = PdfBuilder(minio_client)
    pdf_report_builder.create_report_heading()

    # 1) Extract parameters into model.
    parameters = create_tree_parameters_dto(request, True)
    print_parameters(parameters)

    # 2) Extract Parameters:
    x_train, x_test, y_train, y_test, remaining_x_columns, y_unique, column_names, data, X, Y = prepare_data(parameters.data, parameters.target, parameters.test_size, minio_client, pdf_report_builder)
    dataset_info = Dataset(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test, x_columns=remaining_x_columns, y_unique_values=y_unique, column_names=column_names, is_binary_classification=len(y_unique) > 2, data=data, X=X, Y=Y)
    input_parameters = map_parameters_into_dictionary(parameters)

    # Class Imbalances
    imbalance_detector = ImbalanceDetector()
    imbalance_result = imbalance_detector.analyse_imbalances_in_target_variable(dataset_info, parameters.target)
    pdf_report_builder.append_text_to_report("Data Imbalances", imbalance_result)
    minio_client.upload_file_to_bucket(output_file_name + '/' + imbalance_result, imbalance_result)

    clf = decision_tree_classifier(input_parameters, dataset_info, pdf_report_builder)

    # 3) Interpret Model
    tree_interpreter = TreeInterpreter(clf, dataset_info, minio_client, pdf_report_builder)
    tree_interpreter.interprete_decision_tree(parameters.target)

    TempFileManager.flush_temp_files()


    print()
    print()
    print('=====> PIPELINE EXECUTED <=====')


def get_bucket_folder_from_request(input_string):
    return input_string.split('/')[0]