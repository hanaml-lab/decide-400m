# Toy run (real output)

A real run of the reference implementation on its CPU toy. Unlike the model card's evaluation, which comes from a
seeded simulation and is illustrative, every number below was printed by the code.

- Command: `python -m decide.demo` (settings in `configs/toy.json`)
- Environment: Python 3.13.9, torch 2.12.0, numpy 2.3.5; Linux x86_64, Intel(R) Core(TM) i9-14900KF, 4 threads
- Wall time: 74 s, with the machine shared

## Output

```
CLINC150 toy: 9375 training states, 1875 calibration, 2813 test; 3 questions per state, one pass each; 0.40M parameters

kind      accuracy   ECE raw  ECE cal.      T
choice       0.883     0.033     0.023   1.35
yes/no       0.924     0.031     0.013   1.37
score        0.708     0.016     0.015   1.25

decide or defer at alpha = 0.05, delta = 0.10 (thresholds set on calibration, read on test)
kind     threshold  decided  error among decided
choice       0.885    45.5%                 2.2%
yes/no       0.905    78.4%                 2.6%
score        0.970    15.8%                 2.5%
choice questions no option fits (32% of test): 1.2% decided; options-only softmax, same rule: 1.4% decided, 42.7% coverage, 2.0% error

decide or defer at alpha = 0.10, delta = 0.10 (thresholds set on calibration, read on test)
kind     threshold  decided  error among decided
choice       0.590    58.3%                 6.8%
yes/no       0.500   100.0%                 7.6%
score        0.690    41.5%                 6.9%
choice questions no option fits (32% of test): 4.5% decided; options-only softmax, same rule: 7.5% decided, 55.4% coverage, 6.6% error

options-only softmax that always answers: decides 100%, error 38.5%, and answers every one of the 898 questions no option fits
options shuffled on every test choice: largest change in any probability 6.8e-06
(74 s on CPU, torch 2.12.0+cu126, 4 threads)
```

See the README for what each line shows.
