This repository contains the training, validation and testing  scripts, used in the FOCCUS project, for D6.2. 

The purpose of this repository is to publish the scripts used to predict Suspended Particulate Matter fields from Delft3D-FM hydrodynamic data, using a ConvLSTM model setup.


To Use:

- Install the environment using the ts218_env_requirements.txt, according to your usual venv instructions.

- Use config_multi.yml or config_single.yml to set up your model domain, and data settings. 
  Use _multi.yml for training multiple models with different hyperparameter sets at a time, use _single.yml for only training and testing one set of hyperparamters.
- Run main.py.
- Other scripts will be called in main.py.
  

Example result: 
![spm_map_day_10 -cut](https://github.com/user-attachments/assets/a825a23c-e928-4904-a8f1-15fce812e2a6)

Further validation and accuracy assessments will occur over the duration of WP7.
FOCCUS is funded by the European Union (Grant Agreement No. 101133911). Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the European Health and Digital Executive Agency (HaDEA). Neither the European Union nor the granting authority can be held responsible for them.

Credits: 
Senyang Li, Beau van Koert, Lotta Beyaard
