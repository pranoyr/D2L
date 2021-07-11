from dataset.seti_dataset import SETIDataset
import torch
import torchvision.transforms as transforms


def get_data_loader(args):
    if args.dataset_name == 'Seti-Dataset':
        transform = transforms.Compose([
		transforms.Resize((256,256))])
        train_data = SETIDataset(
            dataset_dir=args.dataset_dir,
            num_classes=1,
            transform = transform
            )
        data_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch, shuffle=True)
    return data_loader