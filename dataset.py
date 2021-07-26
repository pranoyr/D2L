from datasets.seti_dataset import SETIDataset
import torch
import torchvision.transforms as transforms
from datasets.animals_dataset import AnimalsDataset, collate_skip_empty
from config import cfg


def get_data_loader(args):
    if args.dataset_name == 'Seti-Dataset':
        transform = transforms.Compose([
		transforms.Resize((256,256))])
        train_data = SETIDataset(
            dataset_dir=args.dataset_dir,
            num_images = args.num_images,
            num_classes=1,
            transform = transform
            )
        data_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch, shuffle=True)

        cfg.DATASET.COLORS_PER_CLASS = {
                '0' : [254, 202, 87],
                '1' : [255, 107, 107]
        }

    if args.dataset_name == 'Animal-Dataset':
        transform = transforms.Compose([
            transforms.Resize((256,256)),
            # transforms.RandomCrop(32, padding=3),
            # transforms.ColorJitter(brightness=[0.5,0.5]),
            # transforms.RandomRotation(180),
            # GaussianNoise(0.5),
            transforms.ToTensor()
            ])

        train_data = AnimalsDataset(
            dataset_dir=args.dataset_dir,
            num_images=args.num_images,
            transform = transform
            )
        data_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch, collate_fn=collate_skip_empty, shuffle=True)

        cfg.DATASET.COLORS_PER_CLASS = {
            'withmask' : [254, 202, 87],
            'withoutmask' : [255, 107, 107]
        }
    return data_loader