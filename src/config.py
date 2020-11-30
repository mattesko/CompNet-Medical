from pathlib import Path
import os

# Keep this config file under the `src` directory
# Otherwise, you might not point to the project's root directory properly
_config_file_path = Path(__file__).resolve()
_project_dir = os.path.dirname(_config_file_path.parent)
_registrations_dir = os.path.dirname(_config_file_path.parent.parent)

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
    'chaos_registrations': os.path.join(_registrations_dir, 'registrations')
}

class Directories:
    DATA = data_directory
    
    # CompositionalNets github repository directory
    COMPOSITIONAL_NETS = os.path.join(_project_dir, 'CompositionalNets')
    
    # U-Net model checkpoint directory
    CHECKPOINTS =  os.path.join(_project_dir, 'checkpoints')
    
    # Data directories
    LITS =  os.path.join(data_directory, 'lits')
    CHAOS = os.path.join(data_directory, 'chaos')
    CHAOS_REGISTRATIONS = os.path.join(_registrations_dir, 'registrations')
    RSNA = os.path.join(data_directory, 'rsna-pneumonia-detection-challenge')
    SHENZHEN = os.path.join(data_directory, 'shenzhen_cxr')
    PULMONARY_CXR_ABNORMALITIES = os.path.join(data_directory, 'pulmonary_cxr_abnormalities')
    CHESTX_RAY8 = os.path.join(data_directory, 'chestx-ray8')
    BASELINES = os.path.join(data_directory, 'baselines')

# filepaths = {
    
# }
