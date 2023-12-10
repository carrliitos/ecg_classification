import os

def get_context(current_file_path):
    """
    Get the parent directory of the current file.

    Parameters:
    - current_file_path (str): Path to the current file.

    Returns:
    - str: Absolute path of the parent directory.
    """
    # Get the absolute path of the current directory.
    current_dir = os.path.abspath(os.path.join(current_file_path, os.pardir))

    # Get the absolute path of the parent directory.
    parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))

    return parent_dir
