from collections import namedtuple
from pathlib import Path

# Dataset root directory
_DATASET_ROOT = Path(__file__).parent / ''

Dataset = namedtuple('Dataset', ['name', 'root', 'src', 'bug_repo'])

# Source codes and bug repositories

tomcat = Dataset(
    'tomcat',
    _DATASET_ROOT / 'dataset',
    _DATASET_ROOT / 'dataset/source_files/tomcat-7.0.51',
    _DATASET_ROOT / 'dataset/bug_reports/Tomcat.txt'
)

aspectj = Dataset(
    'aspectj',
    _DATASET_ROOT / 'dataset',
    _DATASET_ROOT / 'dataset/source_files/org.aspectj-bug433351/',
    _DATASET_ROOT / 'dataset/bug_reports/AspectJ.txt'
)

swt = Dataset(
    'swt',
    _DATASET_ROOT / 'dataset',
    _DATASET_ROOT / 'dataset/source_files/eclipse.platform.swt-xulrunner-31/',
    _DATASET_ROOT / 'dataset/bug_reports/SWT.txt'
)

eclipse = Dataset(
    'eclipse',
    _DATASET_ROOT / 'dataset',
    _DATASET_ROOT / 'dataset/source_files/eclipse.platform.ui-johna-402445/',
    _DATASET_ROOT / 'dataset/bug_reports/Eclipse_Platform_UI.txt'
)

birt = Dataset(
    'birt',
    _DATASET_ROOT / 'dataset',
    _DATASET_ROOT / 'dataset/source_files/birt-20140211-1400',
    _DATASET_ROOT / 'dataset/bug_reports/Birt.txt'
)

# Current dataset in use. (change this name to change the dataset)
DATASET = tomcat

if __name__ == '__main__':
    print(DATASET.name, DATASET.root, DATASET.src, DATASET.bug_repo)
