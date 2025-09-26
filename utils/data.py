import matplotlib.pyplot as plt

# Useful for sanity check. 
def plot_example_1D(X, y):
    """Plot the first 5 images and their labels in a row."""
    for i, (img, y) in enumerate(zip(X[:5].reshape(5, 28, 28), y[:5])):
        plt.subplot(151 + i)
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        plt.title(y)

# Pytorch preprocessing
achille_preprocess = transforms.Compose([
    transforms.Pad(2),       # 28x28 → 32x32 
    transforms.ToTensor(),
])

achille_transform_train = transforms.Compose([
    transforms.Pad(2), 
    transforms.RandomAffine(degrees=0, translate=(4/32, 4/32)),  # random translation
    transforms.RandomHorizontalFlip(),  # horizontal flip
    transforms.ToTensor(),
])