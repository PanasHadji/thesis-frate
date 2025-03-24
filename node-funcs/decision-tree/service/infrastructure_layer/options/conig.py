import json
import os


class Config:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(Config, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'initialized'):  # Ensure init is run only once
            # Resolve the absolute path to the config.json file
            config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../config.json'))
            with open(config_path, 'r') as config_file:
                config_data = json.load(config_file)

            self.dev_mode = config_data['dev_mode']
            self.use_pva97kn_dataset = config_data['use_pva97kn_dataset']
            self.min_io_test_bucket = config_data['min_io_test_bucket']
            self.use_pandas_data_splitting = config_data['use_pandas_data_splitting']
            self.min_io_endpoint = config_data['min_io_endpoint']
            self.initialized = True


# Create a global instance
_config = Config()

