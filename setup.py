from setuptools import find_packages, setup
from typing import List

# -e . should not be present in the list of requirements
HYPEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    """
    This function will return the list of requirements
    
    Args:
        file_path (str): Path to requirements.txt file
        
    Returns:
        List[str]: List of package requirements
    """
    requirements = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file_obj:
            requirements = file_obj.readlines()
            requirements = [req.replace("\n", "").strip() for req in requirements]
            
            # Remove empty lines and comments
            requirements = [req for req in requirements if req and not req.startswith('#')]
            
            if HYPEN_E_DOT in requirements:
                requirements.remove(HYPEN_E_DOT)
                
    except FileNotFoundError:
        print(f"Warning: {file_path} not found. No requirements will be installed.")
    except Exception as e:
        print(f"Error reading requirements: {e}")
        
    return requirements

setup(
    name='student-performance-predictor',
    version='0.1.0',
    author='Shyam',
    author_email='rkknightlx@gmail.com',
    description='A machine learning application to predict student math scores',
    long_description=open('README.md', 'r', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/DataNinja22/Predict_Marks',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Education',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    python_requires='>=3.8',
    install_requires=get_requirements('requirements.txt'),
    keywords='machine learning, education, prediction, student performance',
    project_urls={
        'Bug Reports': 'https://github.com/DataNinja22/Predict_Marks/issues',
        'Source': 'https://github.com/DataNinja22/Predict_Marks',
    },
)