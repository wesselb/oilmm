`OILMM <http://github.com/wesselb/oilmm>`__
===========================================

|CI| |Coverage Status| |Latest Docs| |Code style: black|

Implementation of the Orthogonal Instantaneous Linear Mixing Model

User Installation With ``pip``
------------------------------

See `the instructions
here <https://gist.github.com/wesselb/4b44bf87f3789425f96e26c4308d0adc>`__.
Then simply

.. code:: bash

    pip install oilmm

Repo Installation Without ``pip``
---------------------------------

See `the instructions
here <https://gist.github.com/wesselb/4b44bf87f3789425f96e26c4308d0adc>`__.

Then clone and enter the repo.

.. code:: bash

    git clone https://github.com/wesselb/oilmm
    cd oilmm

Finally, make a virtual environment and install the requirements.

.. code:: bash

    virtualenv -p python3 venv
    source venv/bin/activate
    pip install -r requirements.txt -e .

Reproduce Experiments
---------------------

Scripts to rerun individual experiments from the paper can be found in
the ``experiments`` folder.

A shell script is provided to rerun all experiments from the paper at
once:

.. code:: bash

    sh run_experiments.sh

The results can then be found in the generated ``_experiments`` folder.

.. |CI| image:: https://github.com/wesselb/oilmm/workflows/CI/badge.svg?branch=master
   :target: https://github.com/wesselb/oilmm/actions?query=workflow%3ACI
.. |Coverage Status| image:: https://coveralls.io/repos/github/wesselb/oilmm/badge.svg?branch=master&service=github
   :target: https://coveralls.io/github/wesselb/oilmm?branch=master
.. |Latest Docs| image:: https://img.shields.io/badge/docs-latest-blue.svg
   :target: https://wesselb.github.io/oilmm
.. |Code style: black| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
