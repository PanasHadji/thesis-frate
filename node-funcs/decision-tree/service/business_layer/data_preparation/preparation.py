import numpy as np
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from service.business_layer.experiment.experiment import split_data_using_pandas
from service.infrastructure_layer.options.conig import _config
from service.infrastructure_layer.constants import State
from service.infrastructure_layer.external.minio import MinIoClient
from service.infrastructure_layer.options.minio_options import MinIoConfig
from service.infrastructure_layer.pdf_reporter import PdfBuilder


def prepare_data(data, target: str, test_size: int, minIOClient: MinIoClient, pdf_builder: PdfBuilder):
    print('START prepare_data')

    null_counts = data.isnull().sum()
    print(null_counts)

    data = convert_comma_decimal(data)

    encode_categorical_columns(data, minIOClient, pdf_builder)

    if _config.use_pva97kn_dataset is True:
        # List of columns to drop
        columns_to_drop = [
            'dempctveterans', 'giftavgcard36', 'demhomeowner', 'demgender',
            'promcntcard36', 'demage', 'giftavgall', 'giftavglast', 'giftcnt36',
            'giftcntall', 'id', 'giftcntcardall', 'promcntcardall', 'promcnt36',
            'promcntcard12', 'promcnt12', 'demcluster', 'demmedincome',
            'statuscatstarall', 'promcntall', 'gifttimelast'
        ]

        # Drop the columns
        data.drop(columns=columns_to_drop, inplace=True)

    column_names = data.columns

    # Get the column names of remaining independent variables
    remaining_x_columns = [column for column in data.columns
                           if column != target]

    # TODO: Handle this Log it as information to the user.
    # Select the target variable column by name
    y = data[target]
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    y_unique = np.unique(y)

    # Select the remaining columns (features) by dropping the target column
    X = data.drop(columns=[target])

    # Split dataset
    if (_config.use_pandas_data_splitting):
        x_train, y_train, x_test, y_test = split_data_using_pandas(data, target)
    else:
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=State.DEFAULT.value)

    return x_train, x_test, y_train, y_test, remaining_x_columns, y_unique, column_names, data, X, y


def prepare_dataset_format(data: DataFrame):
    data.columns = map(str.lower, data.columns)
    return data


def encode_categorical_columns(df, minIoClient: MinIoClient, pdf_builder: PdfBuilder):
    encoded_list = []
    label_encoders = {}
    for column in df.columns:
        if df[column].dtype == 'object' or df[column].dtype.name == 'category':
            label_encoders[column] = LabelEncoder()
            df[column] = label_encoders[column].fit_transform(df[column])
            encoded_title = f"Column '{column}' converted using Label Encoding:"
            encoded_list.append('')
            encoded_list.append('')
            encoded_list.append(encoded_title)
            print(encoded_title)
            for original, encoded in zip(label_encoders[column].classes_,
                                         label_encoders[column].transform(label_encoders[column].classes_)):
                encode_text = f"  '{original}' -> '{str(encoded)}'"
                print(encode_text)
                encoded_list.append(encode_text)

    file_name = "label_encoding.txt"
    with open(file_name, "w", encoding="utf-8") as file:
        for encode_info in encoded_list:
            file.write(encode_info + "\n")
    pdf_builder.append_text_to_report('Label Encodings to handle categorical variables', file_name)
    minIoClient.upload_file_to_bucket(MinIoConfig.get_folder_name() + '/' + file_name, file_name)
    return df


def convert_comma_decimal(df):
    print()
    print()
    print('===> preparation.convert_comma_decimal <===')
    df_converted = df.copy()
    for column in df.columns:
        if df[column].dtype == 'object':
            try:
                df_converted[column] = df[column].str.replace(',', '.').astype(float)
                print(f"Column '{column}' converted to numeric values.")
            except ValueError:
                pass
    return df_converted
