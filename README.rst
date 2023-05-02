============
PyMEGABASE
============

|Citing PyMEGABASE|
|NDB|
|Update|
|GitHub-Stars|

.. |Citing PyMEGABASE| image:: https://img.shields.io/badge/cite-PyMEGABASE-blue
   :target: https://ndb.rice.edu/MEGABASE-Documentation
.. |NDB| image:: https://img.shields.io/badge/NDB-Nucleome%20Data%20Bank-informational
   :target: https://ndb.rice.edu/
.. |Update| image:: https://img.shields.io/github/last-commit/ed29rice/PyMEGABASE  
.. |GitHub-Stars| image:: https://img.shields.io/github/stars/ed29rice/PyMEGABASE?style=social
   :target: https://github.com/ed29rice/PyMEGABASE

Overview
========
PyMEGABASE (PYMB) is Python library for calling subcompartment annotations based on 1D epigenetic tracks.

Resources
=========
`Tutorial prediction of human and mouse cell types: <https://colab.research.google.com/drive/1U5ZNTg8A6tMNsIyHJV7zDFSUThisTH3F?usp=sharing>`



Citation
========

TBD.


Installation
============

The **PyMEGABASE** library can be installed via pip.

- pip install -i https://test.pypi.org/pypi/ --extra-index-url https://pypi.org/simple PyMEGABASE==1.0.13

You may need to install pyBigWig (pip install pyBigWig) before installing PyMEGABASE

The following are libraries **required** for installing **PyMEGABASE**:

- 'numpy'
- 'glob2>=0.7'
- 'requests'
- 'pytest-shutil>=1.7.0'
- 'pyBigWig>=0.3.18'
- 'urllib3>=1.26.14'
- 'tqdm>=4.64.1'
- 'joblib>=1.2.0'
- 'pydca>=1.23'
- 'ipywidgets>=8.0.4'
