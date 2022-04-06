
## Setup: 

Run `pip3 install -r requirements.txt` and it will install the full set of system dependencies. 

## Running the code: 

1.  `dev_notebook.ipynb` has example code for running an experiment and viewing the results. The main steps involved are: 

```
from train import *

params = {
    'lr': 0.05,
    'dp': True,
    'num_microbatches': 32,
    'batch_size': 32,
    'S': 1,
    'z': 1.1,
    'momentum': 0.5,
    'decay': 0,
    'n_epochs' : 40,
}

run_exp('exp_name', params, use_devset=True)
```

Where you can change the parameters to different configurations to try different values. `exp_name` is a string for the specific experiment setting. This will create new files in `runs/{exp_name}` that includes (1) TF Events for real-time visualization, (2) a CSV file with the train and validation loss, and (3) a json file with the experiment parameters. If you set the `use_devset=True` , this will use a much smaller dataset (~50 batches) for testing and debugging purposes. 

2. If you want to visualize loss and accuracy during training, you can use [tensorboardX](https://github.com/lanpa/tensorboardX). Just open the terminal and type `tensorboard --logdir runs` and it will launch a webpage with graphs that update dynamically over epochs.

3. To fetch and view experiment results, call `run, params = helpers.load_exp('exp_name')`, where `exp_name` is the same tag used in step (1) above. This will return a pandas dataframe `run` and a `params`dictionary with the hyperparameter settings used in the experiment. You can make convergence plots and tables based on the dataframe returned. 



## Resources:
-   [pytorch-privacy](https://github.com/ebagdasa/pytorch-privacy): PyTorch implementation of DP-SGD. See `train.py` for a good overview of the process, and `utils/parameters.yaml` for initial hyperparameters to try out