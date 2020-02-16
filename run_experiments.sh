export PYTHONPATH=.
export TK_SILENCE_DEPRECATION=1

python experiments/eeg.py
python experiments/eeg_ilmm.py
python experiments/exchange.py
python experiments/exchange_ilmm.py
python experiments/temperature.py
python experiments/simulators.py
python experiments/simulators_process.py