import torch
import numpy as np
import faiss
from torchvision import transforms
from model import FeatureExtractor
import bisect
from dataset import RParisDataset
from pathlib import Path
from setup import workdir

root = Path(workdir)

transform = transforms.Compose(
    [
        transforms.Resize(
            (224, 224), interpolation=transforms.InterpolationMode.BILINEAR
        ),
        transforms.ToTensor(),
    ]
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


feature_root = root / "roxford-rparis" / "embedding"

resnet_l2_index = faiss.read_index(str(feature_root / "rparis_resnet50.l2_index.bin"))

aug_resnet_l2_index = faiss.read_index(
    str(feature_root / "cropped_rparis_resnet50.l2_index.bin")
)

hog_l2_index = faiss.read_index(str(feature_root / "rparis_hog.l2_index.bin"))

extractor = FeatureExtractor(device)

ds = RParisDataset(transform=None)


def retrieve(query_img, k, index=resnet_l2_index):
    img = transform(query_img)
    img = img.unsqueeze(0).to(device)

    feat = extractor.extract_features(img)
    distance, indices = index.search(feat, k)

    return indices[0].tolist(), distance[0].tolist()


def retrieve_augmented(query_img, k):
    src_indices, src_dst = retrieve(query_img, k, index=resnet_l2_index)
    src_indices, src_dst = extend_answer_on_augment_ds(
        query_img, k * 4, src_indices, src_dst, aug_index=aug_resnet_l2_index
    )
    return src_indices, src_dst


def extend_answer_on_augment_ds(query_im, k, src_indices, src_dst, aug_index):
    aug_indices, aug_dst = retrieve(query_im, k, index=aug_index)
    for idx, aug_ans in enumerate(aug_indices):
        original_idx = aug_ans // 4

        if original_idx not in src_indices:
            ins_pos = bisect.bisect_left(src_dst, aug_dst[idx])
            src_indices.insert(ins_pos, original_idx)
            src_dst.insert(ins_pos, aug_dst[idx])

    return src_indices, src_dst


def get_image_from_index(idx):
    return ds[idx]
