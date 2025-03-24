
from minio import Minio, S3Error
from service.infrastructure_layer.options.minio_options import MinIoConfig
from service.infrastructure_layer.options.conig import _config

class MinIoClient:
    def __init__(self, access_key, secret_key, bucket_name, folder):

        print('MinIoClient instantiated!')
        # Extract configuration from parsed JSON input
        self.access_key = access_key
        self.secret_key = secret_key
        self.bucket_name = bucket_name
        self.client = Minio(
            _config.min_io_endpoint,
            access_key=self.access_key,
            secret_key=self.secret_key,
            secure=False
        )
        MinIoConfig.set_folder_name(folder)
        MinIoConfig.set_access_key(access_key)
        MinIoConfig.set_secret_key(secret_key)
        MinIoConfig.set_bucket_name(bucket_name)

    def read_data_from_picked(self, input_file_name):
        try:
            # Read data from pickled file in MinIO
            obj = self.client.get_object(self.bucket_name, input_file_name)
            pickled_data = obj.read()
            return pickled_data
        except S3Error as exc:
            raise RuntimeError(f"An error occurred while reading from MinIO: {exc}")
        except Exception as exc:
            raise RuntimeError(f"Error processing file: {exc}")

    def upload_to_bucket(self, output_file_name, bytes_buffer):
        try:
            print('upload_to_bucket!')
            self.client.put_object(self.bucket_name, output_file_name, bytes_buffer, len(bytes_buffer.getvalue()))
        except S3Error as exc:
            raise RuntimeError(f"An error occurred while uploading to MinIO: {exc}")
        except Exception as exc:
            raise RuntimeError(f"Error processing file: {exc}")

    def upload_file_to_bucket(self, path, file_name):
        try:
            # file_name example: '3122abacc7b46c0b-f66b4fc2136877d0/dummy.txt'
            print('upload_file_to_bucket!')
            self.client.fput_object(self.bucket_name, path, file_name)
            print(f"'{file_name}' is successfully uploaded to bucket '{self.bucket_name}'.")
        except S3Error as exc:
            raise RuntimeError(f"An error occurred while uploading to MinIO: {exc}")
        except Exception as exc:
            raise RuntimeError(f"Error processing file: {exc}")

        print(f"'{file_name}' has been deleted from the local filesystem.")
