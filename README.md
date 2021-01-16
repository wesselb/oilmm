# [OILMM](http://github.com/wesselb/oilmm)

[![CI](https://github.com/wesselb/oilmm/workflows/CI/badge.svg?branch=master)](https://github.com/wesselb/oilmm/actions?query=workflow%3ACI)
[![Coverage Status](https://coveralls.io/repos/github/wesselb/oilmm/badge.svg?branch=master)](https://coveralls.io/github/wesselb/oilmm?branch=master)
[![Latest Docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://wesselb.github.io/oilmm)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


Implementation of the Orthogonal Instantaneous Linear Mixing Model

## User Installation With `pip`

See [the instructions here](https://gist.github.com/wesselb/4b44bf87f3789425f96e26c4308d0adc).
Then simply

```bash
pip install oilmm
```

## Repo Installation Without `pip`

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
pip install -r requirements.txt -e .
```

## Reproduce Experiments

Scripts to rerun individual experiments from the paper can be found in the
`experiments` folder.

A shell script is provided to rerun all experiments from the paper at once:

```bash
sh run_experiments.sh
```

The results can then be found in the generated `_experiments` folder.

