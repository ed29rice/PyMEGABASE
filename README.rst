============
PyMEGABASE
============

|Citing PyMEGABASE|
|PyPI|
|conda-forge|
|ReadTheDocs|
|NDB|
|Update|
|GitHub-Stars|

.. |Citing PyMEGABASE| image:: https://img.shields.io/badge/cite-OpenMiChroM-informational
   :target: https://open-michrom.readthedocs.io/en/latest/Reference/citing.html
.. |PyPI| image:: https://img.shields.io/pypi/v/OpenMiChroM.svg
   :target: https://pypi.org/project/OpenMiChroM/
.. |conda-forge| image:: https://img.shields.io/conda/vn/conda-forge/OpenMiChroM.svg
   :target: https://anaconda.org/conda-forge/PyMEGABASE
.. |ReadTheDocs| image:: https://readthedocs.org/projects/pymegabase/badge/?version=latest
   :target: https://pymegabase.readthedocs.io/en/latest/
.. |NDB| image:: https://img.shields.io/badge/NDB-Nucleome%20Data%20Bank-informational
   :target: https://ndb.rice.edu/
.. |Update| image:: https://anaconda.org/conda-forge/openmichrom/badges/latest_release_date.svg   
   :target: https://anaconda.org/conda-forge/openmichrom
.. |GitHub-Stars| image:: https://img.shields.io/github/stars/junioreif/OpenMiChroM.svg?style=social
   :target: https://github.com/ed29rice/PyMEGABASE


`Documentation <https://open-michrom.readthedocs.io/>`__
| `Install <https://open-michrom.readthedocs.io/en/latest/GettingStarted/installation.html>`__
| `Tutorials <https://open-michrom.readthedocs.io/en/latest/Tutorials/Tutorial_Single_Chromosome.html>`__
| `Forum <https://groups.google.com/g/open-michrom>`__

Overview
========

`PyMEGABASE <https://www.sciencedirect.com/science/article/pii/S0022283620306185>`_ is a Python library for performing chromatin dynamics simulations and analyses. Open-MiChroM uses the  `OpenMM <http://openmm.org/>`_ Python API employing the `MiChroM (Minimal Chromatin Model) <https://www.pnas.org/content/113/43/12168>`_ energy function. The chromatin dynamics simulations generate an ensemble of 3D chromosomal structures that are consistent with experimental Hi-C maps. OpenMiChroM also allows simulations of a single or multiple chromosome chain using High-Performance Computing in different platforms (GPUs and CPUs).

.. .. raw:: html

..     <p align="center">
..     <img align="center" src="./docs/source/images/OpenMiChroM_intro_small.jpg" height="300px">
..     </p>

Resources
=========

- `Reference Documentation <https://open-michrom.readthedocs.io/>`__: Examples, tutorials, and class details.
- `Installation Guide <https://open-michrom.readthedocs.io/en/latest/GettingStarted/installation.html>`__: Instructions for installing **PyMEGABASE**.
- `Open-MiChroM Google Group <https://groups.google.com/g/open-michrom>`__: Ask questions to the **PyMEGABASE** user community.
- `GitHub repository <https://github.com/junioreif/OpenMiChroM/>`__: Download the **PyMEGABASE** source code.
- `Issue tracker <https://github.com/junioreif/OpenMiChroM/issues>`__: Report issues/bugs or request features.


Citation
========

When using **PyMEGABASE** to perform chromatin dynamics simulations or analyses, please `use this citation
<https://open-michrom.readthedocs.io/en/latest/Reference/citing.html>`__.


Installation
============

The **PyMEGABASE** library can be installed via `conda <https://conda.io/projects/conda/>`_ or pip, or compiled from source.

Install via conda
-----------------

The code below will install **PyMEGABASE** from `conda-forge <https://anaconda.org/conda-forge/PyMEGABASE>`_.

.. code-block:: bash

    conda install -c conda-forge PyMEGABASE

.. hint:: Often, the installation via conda happens to be stuck. If this is the case, it is recommended to update conda/anaconda using the command line below and try to install **PyMEGABASE** again.

.. code-block:: bash

    conda update --prefix /path/to/anaconda3/ anaconda


Install via pip
-----------------

The code below will install **PyMEGABASE** from `PyPI <https://pypi.org/project/PyMEGABASE/>`_.

.. code-block:: bash

    pip install PyMEGABASE
    
The following are libraries **required** for installing **PyMEGABASE**:

- `Python <https://www.python.org/>`__ (>=3.6)
- `NumPy <https://www.numpy.org/>`__ (>=1.14)
- `SciPy <https://www.scipy.org/>`__ (>=1.5.0)
