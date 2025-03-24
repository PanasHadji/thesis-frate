import os


class TempFileManager:
    temp_files = []

    @staticmethod
    def add_temp_file(file_path):
        TempFileManager.temp_files.append(file_path)

    @staticmethod
    def flush_temp_files():
        for temp_file in TempFileManager.temp_files:
            try:
                os.remove(temp_file)
                print(f"Deleted temporary file: {temp_file}")
            except FileNotFoundError:
                print(f"File not found: {temp_file}")
        # Clear the list after cleanup
        TempFileManager.temp_files.clear()