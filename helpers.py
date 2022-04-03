import json
import pandas as pd

def write_logs(exp_name, log, params):
    # Directly from dictionary
    df = pd.DataFrame.from_dict(log)
    df.to_csv(f'runs/{exp_name}.csv', index=False)

    with open(f'runs/{exp_name}_params.json', "w") as outfile:
        json.dump(params, outfile)

def load_exp(exp_name):
    
    mnist, cifar  = pd.read_csv(f'runs/{exp_name}/mnist.csv'), pd.read_csv(f'runs/{exp_name}/cifar.csv')
    mnist['benchmark'] = 'MNIST'
    cifar['benchmark'] = 'CIFAR'
    log_data = pd.concat([mnist, cifar])

    with open(f'runs/{exp_name}/mnist_params.json') as json_file:
        params = json.load(json_file)

    return log_data, params
    