# qnas
Quantum Neural Architecture Search with Bayesian Optimization 

## Running Examples
First make sure this repo directory is on the PYTHONPATH, e.g. by running:
```bash
$ source shell/add_pwd_to_pythonpath.sh
```

And then run example scripts, such as:
```bash
$ python src/main.py -obj qgan -n 3 -no 9 -init 10 -T 10 -B 50 -S 1 -s 27112021 -dir ./output/ --gpuid 7
```
