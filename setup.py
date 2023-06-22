from setuptools import find_packages,setup
from typing import List

E_DOT = '-e .'


def get_requirements(file_path:str)->List[str]:
    '''
    this function will return the list of requirement

    '''

    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n","") for req in requirements]

        if E_DOT in requirements:
            requirements.remove(E_DOT)

    return requirements






setup(
    name = 'chaits_ml_project',
    versions = '0.0.1',
    author = 'Chaitanya',
    author_email = 'ugalechaitanya80@gmail.com',
    packages = find_packages(),
    install_requires = get_requirements('requirements.txt')
)