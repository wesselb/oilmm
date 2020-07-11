export PYTHONPATH=.
export TK_SILENCE_DEPRECATION=1

python experiments/eeg.py
python experiments/eeg_ilmm.py
python experiments/exchange.py
python experiments/exchange_ilmm.py
for m in 1 2 5 10 15 20 25 50 75 100 125 150 175 200 225 247; do
    python experiments/temperature.py -m $m
done
python experiments/temperature_igp.py
python experiments/simulators.py
python experiments/simulators_process.py
