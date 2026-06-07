# Module for importing an dexporting
import numpy as np

def load_csv(file_path, delimiter=',', missing_values='', fill_value=np.nan, skip_header=1):
    """Loads a CSV file into a NumPy array using genfromtxt.
    
    Args:
        file_path (str): Path to the CSV file.
        delimiter (str): The character separating values.
        missing_values (str): The string representing missing data.
        fill_value (float): The value to replace missing data with.
        skip_header (int): Number of lines to skip at the beginning.
        
    Returns:
        np.ndarray: The loaded data array.
    """
    return np.genfromtxt(
        file_path,
        delimiter=delimiter,
        missing_values=missing_values,
        filling_values=fill_value,
        skip_header=skip_header,
        dtype=float # Forces numeric parsing
    )
    