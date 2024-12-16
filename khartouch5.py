import matplotlib.pyplot as plt

# Data for plotting
random_seeds = {
    1: {
        'lambda': [1.0, 10.0, 100.0, 1e7],
        'train_loss': [14.36, 13.89, 13.36, 13.40],
        'test_loss': [13.92, 13.77, 13.02, 13.26]
    },
    2: {
        'lambda': [1.0, 10.0, 100.0, 1e6],
        'train_loss': [13.33, 12.34, 12.28, 12.09],
        'test_loss': [14.39, 13.54, 13.12, 13.04]
    },
    3: {
        'lambda': [1.0, 10.0, 100.0, 1e6],
        'train_loss': [14.17, 13.65, 13.04, 13.29],
        'test_loss': [13.78, 13.31, 15.94, 13.03]
    }
}

# Plotting
plt.figure(figsize=(10, 6))
linestyles = {'train': '-', 'test': '--'}
colors = {1: 'b', 2: 'g', 3: 'r'}

for seed, data in random_seeds.items():
    lambdas = data['lambda']
    plt.plot(lambdas, data['train_loss'], linestyle=linestyles['train'], color=colors[seed], label=f'Seed {seed} Train')
    plt.plot(lambdas, data['test_loss'], linestyle=linestyles['test'], color=colors[seed], label=f'Seed {seed} Test')

plt.xscale('log')
plt.xlabel('Lambda (log scale)')
plt.ylabel('Loss')
plt.title('Training and Test Loss vs Lambda')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()