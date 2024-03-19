import matplotlib.pyplot as plt

def plot_loss(avg_train_losses, avg_val_losses):

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(avg_train_losses, label='Average Train Loss per Epoch')
    plt.plot(avg_val_losses, label='Average Validation Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
