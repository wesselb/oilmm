# [OILMM](http://github.com/wesselb/oilmm)

[![Build](https://travis-ci.org/wesselb/oilmm.svg?branch=master)](https://travis-ci.org/wesselb/oilmm)
[![Coverage Status](https://coveralls.io/repos/github/wesselb/oilmm/badge.svg?branch=master&service=github)](https://coveralls.io/github/wesselb/oilmm?branch=master)
[![Latest Docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://wesselb.github.io/oilmm)

Implementation of the Orthogonal Instantaneous Linear Mixing Model

## Installation

See [the instructions here](https://gist.github.com/wesselb/4b44bf87f3789425f96e26c4308d0adc).

Then clone and enter the repo.

```bash
git clone https://github.com/wesselb/oilmm
cd oilmm
```

Finally, make a virtual environment and install the requirements.

```bash
virtualenv -p python3 venv
source venv/bin/activate
pip install -r requirements_dev.txt
```

## Reproduce Experiments

Scripts to rerun individual experiments from the paper can be found in the
`experiments` folder.

A shell script is provided to rerun all experiments from the paper at once:

```bash
sh run_experiments.sh
```

The results can then be found in the generated `_experiments` folder.

