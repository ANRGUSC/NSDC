# iobt_ns
IoBT Network Synthesis

## Dependencies
- Python >= 3.7
- Python Dependencies:
  - networkx 
  - numpy 
  - this [HEFT](https://en.wikipedia.org/wiki/Heterogeneous_Earliest_Finish_Time) implementation: https://github.com/mackncheesiest/heft

## Installation
Requires Python 3.7 and a recent version of pip. All other dependencies are installed automatically with the package:

```bash
git clone git@github.com:ANRGUSC/iobt_ns.git
pip install -e ./iobt # -e flag optional - useful for development so you don't have to reinstall all the time
```

## Running 
After installation, you can run any of the test files:

```bash
cd test
python ./test_brute_force.py
```
