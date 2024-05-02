import pickle
from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path
from setup import workdir

data_path = Path(workdir) / "roxford-rparis"


def img_name_to_path(file_name):
    landmark = file_name.split("_")[1]
    return data_path / "paris" / landmark / f"{file_name}.jpg"


class RParisDataset(Dataset):
    def __init__(self, transform=None):
        self.transform = transform

        with open("data/gnd_rparis6k.pkl", "rb") as f:
            self.gnd = pickle.load(f)

    def __len__(self):
        return len(self.gnd["imlist"])

    def __getitem__(self, idx):
        img_name = self.gnd["imlist"][idx]

        img_path = img_name_to_path(img_name)

        _img = Image.open(img_path)
        _img = _img.convert("RGB")

        if self.transform:
            _img = self.transform(_img)

        return _img, str(img_name)
