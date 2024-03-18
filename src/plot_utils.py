import matplotlib.pyplot as plt

def plot_loss(loss):
    # Unpack loss
    loss = avg_train_losses_per_epoch, avg_val_losses_per_epoch = loss
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(avg_train_losses_per_epoch, label='Average Train Loss per Epoch')
    plt.plot(avg_val_losses_per_epoch, label='Average Validation Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
