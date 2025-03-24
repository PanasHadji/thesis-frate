class MinIoConfig:
    # Static member properties
    access_key = None
    secret_key = None
    bucket_name = None
    folder_name = None

    @classmethod
    def set_access_key(cls, key):
        cls.access_key = key

    @classmethod
    def get_access_key(cls):
        return cls.access_key

    @classmethod
    def set_secret_key(cls, key):
        cls.secret_key = key

    @classmethod
    def get_secret_key(cls):
        return cls.secret_key

    @classmethod
    def set_bucket_name(cls, name):
        cls.bucket_name = name

    @classmethod
    def get_bucket_name(cls):
        return cls.bucket_name

    @classmethod
    def set_folder_name(cls, name):
        cls.folder_name = name

    @classmethod
    def get_folder_name(cls):
        return cls.folder_name