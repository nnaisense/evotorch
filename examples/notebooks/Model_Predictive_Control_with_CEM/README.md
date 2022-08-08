# Model Predictive Control with CEM

This is the model predictive control (MPC) example which uses the cross entropy method (CEM) implementation of EvoTorch to solve the gym environment `Reacher-v4`.

This example consists of the following files:

- **[reacher_mpc.ipynb](reacher_mpc.ipynb):** the main notebook demonstrating how EvoTorch can be used for MPC.
- **[reacher_model.pickle](reacher_model.pickle):** the pickle file which contains a pre-trained forward model for `Reacher-v4`. This pickle file is used by the main notebook.
- **[train_forward_model/reacher_train.ipynb](train_forward_model/reacher_train.ipynb):** a supplementary notebook provided for completeness, showing the steps used for training the forward model.
