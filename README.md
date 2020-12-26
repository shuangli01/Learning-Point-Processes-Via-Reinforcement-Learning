# Learning-Point-Processes-Via-Reinforcement-Learning
code of paper "Learning Temporal Point Processes Via Reinforcement Learning", NeurIPS 2018.

ppgen.py -- generate synthetic temporal point process data as input data for the demo

tpp_model_policy_gradient.py -- this version of algorithm uses "policy gradient" method to learn the model

tpp_model_Repara_trick.py -- this version of improved algorithm uses reparameterization trick to learn the model (converge faster than the "policy gradient" method) 

policy_gradient.png -- this figure demonstrates the final learning results obtained by policy gradient method

rapara_trick.png -- this figure demonstrates the final learning results obtained by the reparameterization trick method
