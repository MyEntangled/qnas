# qnas
Quantum Neural Architecture Search with Bayesian Optimization 

## Running Examples
First make sure this repo directory is on the PYTHONPATH, e.g. by running:
```bash
$ source shell/add_pwd_to_pythonpath.sh
```

To test if the entire program works
```bash
$ python src/main.py -obj qft -n 2 -no 6 -init 5 -T 2 -B 2 -S 1 -s 6789 -dir output --gpuid 0
```

And then run example scripts, such as:
```bash
$ python src/main.py -obj qgan -n 3 -no 12 -init 5 -T 3 -B 25 -S 1 -s 6789 -dir output --gpuid 0
```
```bash
$ python src/main.py -obj qft -n 3 -no 14 -init 5 -T 3 -B 25 -S 1 -s 6789 -dir output --gpuid 0
```
```bash
$ python src/main.py -obj maxcut -n 9 -no 5 -init 5 -T 3 -B 25 -S 1 -s 6789 -dir output --gpuid 0
```
