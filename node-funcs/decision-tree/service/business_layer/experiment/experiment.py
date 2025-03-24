from pandas import DataFrame

from service.infrastructure_layer.constants import State


def split_data_using_pandas(df : DataFrame, target_column):
    train_df = df.sample(frac=0.8, random_state=State.DEFAULT.value)
    test_df = df.drop(train_df.index)

    x_train = train_df.drop(target_column, axis=1)
    y_train = train_df[target_column]
    x_test = test_df.drop(target_column, axis=1)
    y_test = test_df[target_column]

    return x_train, y_train, x_test, y_test
