import torch



def mnist():
    """Return train and test dataloaders for MNIST."""
    # exchange with the corrupted mnist dataset
    train_images,  train_labels = [], []
    for i in range(5):
        train_images.append(torch.load('../../../data/corruptmnist/train_images_{}.pt'.format(i)))
        train_labels.append(torch.load('../../../data/corruptmnist/train_target_{}.pt'.format(i)))
    train_images = torch.cat(train_images)
    train_labels = torch.cat(train_labels)

    test_images = torch.load('../../../data/corruptmnist/test_images.pt')
    test_labels = torch.load('../../../data/corruptmnist/test_target.pt')

    #print(f"{train_images.shape = }")
    train_images = train_images.unsqueeze(1)
    test_images = test_images.unsqueeze(1)
    #print(f"{train_images.shape = }")

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_images, train_labels),
        batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(  
        torch.utils.data.TensorDataset(test_images, test_labels),
        batch_size=64, shuffle=True)

    return train_loader, test_loader
