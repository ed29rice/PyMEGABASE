.. _installation:

============
Installation
============

The **PyMEGABASE** library can be installed via `conda <https://conda.io/projects/conda/>`_ or `pip <https://pypi.org/>`_, or compiled from `source (GitHub) <https://github.com/ed29rice/PyMEGABASE>`_.

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

    pip3 install PyMEGABASE

.. note::

The following are libraries **required** for installing **PyMEGABASE**:

- `Python <https://www.python.org/>`__ (>=3.6)
- `NumPy <https://www.numpy.org/>`__ (>=1.14)
- `SciPy <https://www.scipy.org/>`__ (>=1.5.0)
