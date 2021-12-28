from jittor.dataset.dataset import ImageFolder
import jittor.transform as transform

transform = transform.Compose([
    transform.Resize(size=[112,112]),
    transform.ImageNormalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])
train_dir = 'E:\NN\GANSketching-main\data\image\cat'
train_loader = ImageFolder(train_dir).set_attrs(transform=transform, batch_size=4, shuffle=True)
val_dir = 'E:\NN\GANSketching-main\data\image\cat'
val_loader = ImageFolder(val_dir).set_attrs(transform=transform, batch_size=1, shuffle=True)