import numpy as np
import torch
from torch.utils import data


def make_dataset(train_val_dist):
		# data = pd.read_csv(csv_file)
	data = train_val_dist.values.tolist()
	labels = np.array([i[1] for i in data])
	class_one_count = len(np.argwhere(labels==1).squeeze(1))
	class_zero_count = len(np.argwhere(labels==0).squeeze(1))

	class_counts = [class_zero_count, class_one_count]   # [0.1,1]
	num_samples = sum(class_counts)

	class_weights = [num_samples/class_counts[i] for i in range(len(class_counts))]
	weights = [class_weights[labels[i]] for i in range(int(num_samples))]
	return data, weights


class SETIDataset(data.Dataset):
	'Characterizes a dataset for PyTorch'

	def __init__(self, dataset_dir, num_classes=1, transform=None):
		'Initialization'
		self.transform = transform
		self.num_classes = num_classes
		self.root_dir = dataset_dir + '/train'
		train_csv = dataset_dir + '/train_labels.csv'
		self.data, _ = make_dataset(train_csv)

	def __len__(self):
		'Denotes the total number of samples'
		return len(self.data)

	def __getitem__(self, index):
		'Generates one sample of data'
		# ---- Get Inputs ----
		x = np.load(
			f"{self.root_dir}/{self.data[index][0][0]}/{self.data[index][0]}.npy")

		label = self.data[index][1]
		if self.transform:
			x = self.transform(torch.from_numpy(x).type(torch.FloatTensor))

		dict_data = {
            'image' : x,
            'label' : str(label),
            'image_path' : None
        }
		return dict_data
