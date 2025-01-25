from setuptools import setup,find_packages

HYPER_E_DOT = "e ."
def get_packages(file_path):
    
    with open(file_path,'rb') as file_obj:
        
        requirements = file_obj.readlines()
        requirements = [req.replace("\n","") for req in requirements]
        
        if  HYPER_E_DOT in requirements:
            requirements.remove(HYPER_E_DOT)
            
setup(
    name = "Tyroid detection Project",
    author = "Aziz Ashfak",
    author_email = "azizashfak@gmail.com",
    description = "This an end to end project ",
    packages = find_packages(),
    install_requires = get_packages('requirements.txt')
    )
        