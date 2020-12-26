# Learning-Point-Processes-Via-Reinforcement-Learning
Code of paper "Learning Temporal Point Processes Via Reinforcement Learning", NeurIPS 2018.
We also provide the code for the Spatio-Temporal Point Processes extension. 

(1) ppgen.py -- generate synthetic temporal point process data as input data for the demo

(2) tpp_model_policy_gradient.py -- this version of algorithm uses "policy gradient" method to learn the model

(3) tpp_model_Repara_trick.py -- this version of improved algorithm uses reparameterization trick to learn the model (converge faster than the "policy gradient" method) 

(4) policy_gradient.png -- this figure demonstrates the final learning results obtained by policy gradient method

(5) rapara_trick.png -- this figure demonstrates the final learning results obtained by the reparameterization trick method
