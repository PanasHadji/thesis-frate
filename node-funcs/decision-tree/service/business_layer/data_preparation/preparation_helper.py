from dataclasses import asdict

from service.business_layer.dtos.decision_tree_classifier_inputs import DecisionTreeClassifierInputs
from service.business_layer.dtos.tree_parameter_dto import TreeParametersDTO


def map_parameters_into_dictionary(params: TreeParametersDTO):
    # Convert the data classes to dictionaries
    default_params = DecisionTreeClassifierInputs()
    default_params_dict = asdict(default_params)
    custom_params_dict = asdict(params)

    # Remove 'target' and 'data' from the custom parameters dictionary
    custom_params_dict.pop('target', None)
    custom_params_dict.pop('data', None)

    # Replace None values in custom_params with default values from default_params
    merged_params_dict = {
        key: custom_params_dict.get(key) if custom_params_dict.get(key) is not None else default_params_dict[key]
        for key in default_params_dict}

    # Create a new DecisionTreeClassifierInputs object with the merged parameters
    #TODO: TODO:ACH merged_params = DecisionTreeClassifierInputs(**merged_params_dict)

    return merged_params_dict