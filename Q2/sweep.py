# sweep.py

import wandb
import train  # Imports train.py
import yaml   # <-- Import YAML

# 1. Define the Sweep configuration (the search space)
sweep_config = {
    'method': 'bayes', 
    'metric': {
        'name': 'val_loss',
        'goal': 'minimize'
    },
    'parameters': {
        'learning_rate': {
            'values': [1e-4, 3e-4, 1e-3, 3e-3]
        },
        'batch_size': {
            'values': [32, 64, 128]
        },
        'num_filters_nf': {
            'values': [8, 16, 32]
        },
        'kernel_size': {
            'values': [3, 5]
        },
        'optimizer': {
            'values': ['adam', 'sgd']
        }
        # 'epochs' will be taken from the config.yaml default
    }
}

def main():
    # Load the project/entity info from the YAML
    with open("config.yaml", 'r') as f:
        base_config = yaml.safe_load(f)
        
    project_name = base_config.get('project_name', 'smai_a4_q2_colorization')
    entity = base_config.get('entity', None)

    # 2. Initialize the Sweep
    sweep_id = wandb.sweep(
        sweep=sweep_config,
        project=project_name,
        entity=entity
    )
    
    print(f"Sweep created with ID: {sweep_id}")
    print("Run the following command to start a sweep agent:")
    print(f"wandb agent {entity or 'YOUR_ENTITY'}/{project_name}/{sweep_id}")

    # 3. Start the agent
    # The agent will call train.run_training
    # train.run_training will load config.yaml as defaults
    # The agent will then provide the overriding sweep parameters
    #
    # To run 20 runs from this script:
    # wandb.agent(sweep_id, function=train.run_training, count=20)

if __name__ == "__main__":
    main()