from setuptools import setup, find_packages
from os import path

this_dir = path.abspath(path.dirname(__file__))
with open(path.join(this_dir, "README.rst")) as f:
    long_description = f.read()

__version__ = "1.0.0"
for line in open(path.join("PyMEGABASE", "__init__.py")):
    if line.startswith("__version__"):
        exec(line.strip())

setup(
    name="PyMEGABASE",
    version=__version__,
    description="PYMEGABASE for chromatin (sub)compartment annotation prediction",
    url="https://ndb.rice.edu/MEGABASE-Documentation",
    author="Esteban Dodero-Rojas",
    author_email="ed29@rice.edu",
    classifiers=[
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        'Programming Language :: Python :: 3.4',
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Natural Language :: English",
    ],
    include_package_data=True,
    packages=find_packages(),
    install_requires=['numpy', 'glob', 'requests', 'shutil', 'pyBigWig', 'urllib','gzip','tqdm','joblib','pydca'],
    entry_points={"console_scripts": ["CLINAME=PyMEGABASE._cli:main"]},
    zip_safe=True,
    long_description=long_description,
    long_description_content_type="text/x-rst",
)