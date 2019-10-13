from .imports import *

import urllib.request

# Get recommended directories for storing datasets (data_dir) and intermediate files generated during training
# (work_dir).
# :param root_dir: Root directory, which is often the Google Cloud Storage bucket when using TPUs.
# :param project: Name of the project.
# :return: Data directory for storaing datasets, and work directory for storing intermediate files.
    
def get_project_dirs(root_dir: str, project: str) -> Tuple[str, str]:
    data_dir: str = os.path.join(root_dir, 'data', project)
    work_dir: str = os.path.join(root_dir, 'work', project)
    gfile.makedirs(data_dir)
    gfile.makedirs(work_dir)
    return data_dir, work_dir

def test_method():
    print("Calling io package of data-science-utils")
