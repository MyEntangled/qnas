# qnas
Quantum Neural Architecture Search with Bayesian Optimization 

## Running Examples
First make sure this repo directory is on the PYTHONPATH, e.g. by running:
```bash
$ source shell/add_pwd_to_pythonpath.sh
```

And then run example scripts, such as:
```bash
$ python src/main.py -obj qgan -n 3 -no 12 -init 5 -T 5 -B 30 -S 1 -s 6789 -dir ./output/ --gpuid 7
```
```bash
$ python src/main.py -obj qft -n 3 -no 14 -init 5 -T 5 -B 30 -S 1 -s 6789 -dir ./output/ --gpuid 7
```
