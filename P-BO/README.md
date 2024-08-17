# Efficient Black-box Adversarial Attacks via Bayesian Optimization Guided by a Function Prior

This repository contains the code for *Efficient Black-box Adversarial Attacks via Bayesian Optimization Guided by a Function Prior*, to ICML 2024.

# Prerequisites

pip install -r requirements.txt

# Attack CIFAR-10 models

python attack.py --model densenet121 --surrogate-model wideresnet --method bo --bo-scale adapt