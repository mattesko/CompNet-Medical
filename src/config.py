from pathlib import Path
import os
import sys

# Keep this config file under the `src` directory
# Otherwise, you might not point to the project's root directory properly
_config_file_path = Path(__file__).resolve()
_project_dir = os.path.dirname(_config_file_path.parent)

# Change the path where you store your data with respect to the project's root
# directory here
data_directory = os.path.join(_project_dir, 'data')

# Change the directory path for your datasets, checkpoints, or 
# CompositionalNets repository here
directories = {
    'data': data_directory,
    'CompositionalNets': os.path.join(_project_dir, 'CompositionalNets'),
    'checkpoints': os.path.join(_project_dir, 'checkpoints'),
    'lits': os.path.join(data_directory, 'lits'),
    'chaos': os.path.join(data_directory, 'chaos'),
    'rsna': os.path.join(data_directory, 'rsna-pneumonia-detection-challenge'),
    'shenzhen': os.path.join(data_directory, 'shenzhen_cxr'),
    'pulmonary_cxr_abnormalities': os.path.join(data_directory, 'pulmonary_cxr_abnormalities'),
    'chestx-ray8': os.path.join(data_directory, 'chestx-ray8'),
}

# filepaths = {
    
# }
