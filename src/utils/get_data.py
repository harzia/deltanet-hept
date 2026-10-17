from torch_geometric.loader import DataLoader
from datasets import Tracking, TrackingTransform, Pileup, PileupTransform, ShardedPointDataset


def get_data_loader(dataset, idx_split, batch_size, num_workers=0):
    train_loader = DataLoader(
        dataset[idx_split["train"]],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    valid_loader = DataLoader(
        dataset[idx_split["valid"]],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        dataset[idx_split["test"]],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return {"train": train_loader, "valid": valid_loader, "test": test_loader}


def get_dataset(dataset_name, data_dir, use_sharded=False):
    # The transform is applied identically whether the data is served from the
    # monolithic in-memory file or from per-event shards, so results match.
    transform = TrackingTransform() if "tracking" in dataset_name else PileupTransform()
    if use_sharded:
        dataset = ShardedPointDataset(data_dir, dataset_name=dataset_name, transform=transform)
    elif "tracking" in dataset_name:
        dataset = Tracking(data_dir, transform=transform, dataset_name=dataset_name)
    elif dataset_name == "pileup":
        dataset = Pileup(data_dir, transform=transform)
    else:
        raise NotImplementedError
    dataset.dataset_name = dataset_name
    return dataset
