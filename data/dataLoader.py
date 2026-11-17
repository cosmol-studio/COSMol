import torch.multiprocessing
from lightning.pytorch.utilities.combined_loader import CombinedLoader
from torch.utils.data import (
    BatchSampler,
    ConcatDataset,
    RandomSampler,
    SequentialSampler,
)
from torch_geometric.loader import DataLoader

torch.multiprocessing.set_sharing_strategy("file_system")


class NodeBudgetBatchSampler(BatchSampler):
    """BatchSampler with Lightning-compatible sampler injection.

    ``batch_size`` is used as a node budget, not as a fixed number of examples.
    The constructor follows PyTorch's BatchSampler API so Lightning can replace
    ``sampler`` with a DistributedSampler under DDP.
    """

    def __init__(self, sampler, batch_size: int, drop_last: bool) -> None:
        super().__init__(sampler, batch_size=batch_size, drop_last=drop_last)
        self.dataset = getattr(
            sampler, "dataset", getattr(sampler, "data_source", None)
        )
        if self.dataset is None:
            raise ValueError(
                "NodeBudgetBatchSampler requires a sampler with a dataset."
            )

    def __iter__(self):
        batch = []
        current_nodes = 0
        for idx in self.sampler:
            if hasattr(self.dataset, "get_num_nodes"):
                num_nodes = self.dataset.get_num_nodes(idx)
            else:
                data = self.dataset[idx]
                num_nodes = data.num_nodes
            if batch and current_nodes + num_nodes > self.batch_size:
                yield batch
                batch = []
                current_nodes = 0
            batch.append(idx)
            current_nodes += num_nodes
            if current_nodes >= self.batch_size:
                yield batch
                batch = []
                current_nodes = 0
        if batch and not self.drop_last:
            yield batch

    def __len__(self):
        # Exact length depends on sample order and graph sizes. This upper bound
        # keeps Lightning and CombinedLoader length queries defined.
        return len(self.sampler)


class KeyedCycleCombinedLoader(CombinedLoader):
    """max_size_cycle variant whose epoch length follows one named loader.

    Lightning's max_size_cycle uses max(len(loader_i)). This loader instead
    stops at len(iterables[main_key]): shorter loaders are cycled by the parent
    CombinedLoader, while longer shuffled loaders are consumed only up to the
    main loader length for the current epoch.
    """

    def __init__(self, iterables, main_key, mode="max_size_cycle") -> None:
        if not isinstance(iterables, dict):
            raise TypeError("KeyedCycleCombinedLoader expects a top-level dict.")
        if main_key not in iterables:
            raise ValueError(
                f"main_key={main_key!r} is not in iterables: {list(iterables)}"
            )
        if mode != "max_size_cycle":
            raise ValueError("KeyedCycleCombinedLoader only supports max_size_cycle.")
        super().__init__(iterables, mode=mode)
        self.main_key = main_key
        self.main_index = list(iterables).index(main_key)
        self.main_loader = iterables[main_key]
        self.main_dataset = getattr(self.main_loader, "dataset", None)
        self.main_length = len(iterables[main_key])
        self._num_yielded = 0

    def _effective_length(self):
        if self.limits is None:
            return self.main_length
        return int(min(self.main_length, self.limits[self.main_index]))

    def __iter__(self):
        self._num_yielded = 0
        return super().__iter__()

    def __next__(self):
        if self._num_yielded >= self._effective_length():
            raise StopIteration
        batch = super().__next__()
        self._num_yielded += 1
        return batch

    def __len__(self):
        return self._effective_length()

    def _dataset_length(self):
        if self.main_dataset is not None:
            return len(self.main_dataset)
        return self.main_length


def dataLoader(
    dataset_name,
    split,
    dataset_path,
    dataset_args={},
    persistent_workers=True,
    batch_size=-1,
    test_set_similarly=None,
    Batch=True,
    sample_base_dist=True,
):
    from models.utils import resolve_path

    dataset_path = resolve_path(dataset_path)

    if dataset_name == "JointPocketPrediction":
        pocket_loader, _, _ = dataLoader(
            dataset_args.pocket.name,
            split,
            dataset_args.pocket.path,
            dataset_args.pocket,
            persistent_workers=dataset_args.pocket.get("num_workers", 16) > 0,
            Batch=Batch,
            sample_base_dist=sample_base_dist,
        )
        pocket_ranker_args = dataset_args.pocket_ranker
        pocket_ranker_loader, _, _ = dataLoader(
            pocket_ranker_args.name,
            split,
            pocket_ranker_args.path,
            pocket_ranker_args,
            persistent_workers=pocket_ranker_args.get("num_workers", 16) > 0,
            Batch=Batch,
            sample_base_dist=sample_base_dist,
        )
        return (
            CombinedLoader(
                {"pocket": pocket_loader, "pocket_ranker": pocket_ranker_loader},
                mode=dataset_args.get("combined_mode", "max_size_cycle"),
            ),
            None,
            None,
        )

    if dataset_name in ["JointDockingConformation"]:
        conf_loader, _, _ = dataLoader(
            dataset_args.conf.name,
            split,
            dataset_args.conf.path,
            dataset_args.conf,
            persistent_workers=dataset_args.conf.get("num_workers", 16) > 0,
            Batch=Batch,
            sample_base_dist=sample_base_dist,
        )
        dock_loader, _, _ = dataLoader(
            dataset_args.dock.name,
            split,
            dataset_args.dock.path,
            dataset_args.dock,
            persistent_workers=dataset_args.dock.get("num_workers", 16) > 0,
            Batch=Batch,
            sample_base_dist=sample_base_dist,
        )
        return (
            CombinedLoader(
                {"conf": conf_loader, "dock": dock_loader},
                mode=dataset_args.get("combined_mode", "max_size_cycle"),
            ),
            None,
            None,
        )

    if dataset_name == "Joint_all":
        pocket_loader, _, _ = dataLoader(
            dataset_args.pocket.name,
            split,
            dataset_args.pocket.path,
            dataset_args.pocket,
            persistent_workers=dataset_args.pocket.get("num_workers", 16) > 0,
            Batch=Batch,
            sample_base_dist=sample_base_dist,
        )
        pocket_ranker_loader, _, _ = dataLoader(
            dataset_args.pocket_ranker.name,
            split,
            dataset_args.pocket_ranker.path,
            dataset_args.pocket_ranker,
            persistent_workers=dataset_args.pocket_ranker.get("num_workers", 16) > 0,
            Batch=Batch,
            sample_base_dist=sample_base_dist,
        )
        conf_loader, _, _ = dataLoader(
            dataset_args.conf.name,
            split,
            dataset_args.conf.path,
            dataset_args.conf,
            persistent_workers=dataset_args.conf.get("num_workers", 16) > 0,
            Batch=Batch,
            sample_base_dist=sample_base_dist,
        )
        dock_loader, _, _ = dataLoader(
            dataset_args.dock.name,
            split,
            dataset_args.dock.path,
            dataset_args.dock,
            persistent_workers=dataset_args.dock.get("num_workers", 16) > 0,
            Batch=Batch,
            sample_base_dist=sample_base_dist,
        )
        combined_loader = KeyedCycleCombinedLoader(
            {
                "conf": conf_loader,
                "dock": dock_loader,
                "pocket": pocket_loader,
                "pocket_ranker": pocket_ranker_loader,
            },
            main_key=dataset_args.get("main_loader"),
            mode=dataset_args.get("combined_mode"),
        )

        return combined_loader, None, None

    num_workers = dataset_args.get("num_workers", 16)

    if batch_size == -1:
        batch_size = dataset_args.get("batch_size")

    y_mean = None
    y_std = None

    seed = dataset_args.get("seed", 95028)



    if dataset_name == "GEOM_SDEGen":
        from data.GEOM_SDEGen import GEOM_SDEGen

        radius_cutoff = dataset_args.get("radius_cutoff")
        remove_hydrogens = dataset_args.get("remove_hydrogens", False)
        sigma = dataset_args.get("sigma", 0.0)
        time_high = dataset_args.get("time_high", 0.9999)
        terminal_time_threshold = dataset_args.get("terminal_time_threshold", 0.8)
        terminal_x0_noise_std = dataset_args.get("terminal_x0_noise_std", 0.0)
        molecule_backend = dataset_args.get("molecule_backend", "cosmolkit")
        dataset = GEOM_SDEGen(
            dataset_path,
            split=split,
            radius_cutoff=radius_cutoff,
            mode="train",
            remove_hydrogens=remove_hydrogens,
            time_high=time_high,
            terminal_time_threshold=terminal_time_threshold,
            terminal_x0_noise_std=terminal_x0_noise_std,
            molecule_backend=molecule_backend,
        )




    if dataset_name == "PDBbind2020":
        from data.PDBbind2020 import PDBbind2020

        radius_cutoff = dataset_args.get("radius_cutoff")
        max_translation_shift = dataset_args.get("max_translation_shift")
        chirality_prior_max_attempts = dataset_args.get(
            "chirality_prior_max_attempts", 32
        )
        chirality_prior_margin = dataset_args.get("chirality_prior_margin", 0.02)
        time_high = dataset_args.get("time_high", 0.9999)
        terminal_time_threshold = dataset_args.get("terminal_time_threshold", 0.8)
        terminal_x0_noise_std = dataset_args.get("terminal_x0_noise_std", 2.0)

        dataset = PDBbind2020(
            dataset_path,
            split,
            radius_cutoff=radius_cutoff,
            mode="train",
            max_translation_shift=max_translation_shift,
            chirality_prior_max_attempts=chirality_prior_max_attempts,
            chirality_prior_margin=chirality_prior_margin,
            prior_extra_noise_std=dataset_args.get("prior_extra_noise_std", 0.0) if split == "train" else 0.0,
            eval_seed=dataset_args.get("eval_seed", 95028),
            time_high=time_high,
            terminal_time_threshold=terminal_time_threshold,
            terminal_x0_noise_std=terminal_x0_noise_std,
        )

    if dataset_name == "PDBbind2020_pocket":
        from data.PDBbind2020_pocket import PDBbind2020_pocket

        train_noise_std = dataset_args.get("train_noise_std", 0.0)
        radius_cutoff = dataset_args.get("radius_cutoff")
        task = dataset_args.get("task", "pocket")

        dataset = PDBbind2020_pocket(
            dataset_path,
            split,
            radius_cutoff=radius_cutoff,
            train_noise_std=train_noise_std,
            mode="train",
            task=task,
            ranker_num_centers=dataset_args.get("ranker_num_centers", 300),
            ranker_noise_std=dataset_args.get("ranker_noise_std", 6.0),
            ranker_oversample_factor=dataset_args.get("ranker_oversample_factor", 5),
        )

    if dataset_name == "Sc_pdb_pocket":
        from data.Sc_pdb_pocket import Sc_pdb_pocket

        radius_cutoff = dataset_args.get("radius_cutoff")
        radius_min = dataset_args.get("radius_min", 0.0)
        train_noise_std = dataset_args.get("train_noise_std", 0.0)
        task = dataset_args.get("task", "pocket")

        dataset = Sc_pdb_pocket(
            dataset_path,
            split,
            radius_cutoff=radius_cutoff,
            radius_min=radius_min,
            train_noise_std=train_noise_std,
            mode="train",
            task=task,
            ranker_num_centers=dataset_args.get("ranker_num_centers", 300),
            ranker_noise_std=dataset_args.get("ranker_noise_std", 6.0),
            ranker_oversample_factor=dataset_args.get("ranker_oversample_factor", 5),
        )



    if dataset_name == "PDBbind_score":
        from data.PDBbind_score import PDBbind_score

        radius_cutoff = dataset_args.get("radius_cutoff")
        remove_hydrogens = dataset_args.get("remove_hydrogens", True)

        dataset = PDBbind_score(
            dataset_path,
            split,
            radius_cutoff=radius_cutoff,
        )

    if dataset_name == "PDBbind_score_negative":
        from data.PDBbind_score import PDBbind_score
        from data.PDBbind_score_negative import PDBbind_score_negative

        radius_cutoff = dataset_args.get("radius_cutoff")

        dataset_0 = PDBbind_score(
            dataset_path,
            split,
            radius_cutoff=radius_cutoff,
            excluded_indices=dataset_args.get("excluded_indices_by_split", {}).get(split, []),
            pose_sampling=dataset_args.get("pose_sampling", "uniform"),
        )

        dataset_1 = PDBbind_score_negative(
            dataset_path,
            split,
            radius_cutoff=radius_cutoff,
            excluded_indices=dataset_args.get("excluded_indices_by_split", {}).get(split, []),
        )

        positive_repeats = dataset_args.get("positive_repeats", 1) if split == "train" else 1
        if type(positive_repeats) is not int or positive_repeats < 1:
            raise ValueError("positive_repeats must be a positive integer")
        dataset = ConcatDataset([dataset_0] * positive_repeats + [dataset_1])

    if Batch:
        max_nodes = dataset_args.get("max_nodes", None)
        max_edges = dataset_args.get("max_edges", None)
        if max_nodes is not None or max_edges is not None:
            if max_edges is not None:
                raise ValueError(
                    "max_edges is not supported with DDP-safe batching yet."
                )
        max_nodes = dataset_args.get("max_nodes", None)
        max_edges = dataset_args.get("max_edges", None)
        if max_nodes is not None or max_edges is not None:
            if max_edges is not None:
                raise ValueError(
                    "max_edges is not supported with DDP-safe batching yet."
                )
            sampler = (
                RandomSampler(dataset)
                if split == "train"
                else SequentialSampler(dataset)
            )
            batch_sampler = NodeBudgetBatchSampler(
                sampler,
                batch_size=max_nodes,
                drop_last=(split == "train"),
            )
            loader_kwargs = {
                "batch_sampler": batch_sampler,
                "num_workers": num_workers,
                "pin_memory": dataset_args.get("pin_memory", True),
                "timeout": dataset_args.get("timeout", 0),
            }
        else:
            loader_kwargs = {
                "batch_size": batch_size,
                "num_workers": num_workers,
                "drop_last": (split == "train"),
                "shuffle": (split == "train"),
                "pin_memory": dataset_args.get("pin_memory", True),
                "timeout": dataset_args.get("timeout", 0),
            }
        if num_workers > 0:
            loader_kwargs["prefetch_factor"] = dataset_args.get("prefetch_factor", 2)
            loader_kwargs["persistent_workers"] = persistent_workers

        if num_workers > 0 and dataset_args.get("multiprocessing_context"):
            loader_kwargs["multiprocessing_context"] = dataset_args["multiprocessing_context"]
        loader = DataLoader(dataset, **loader_kwargs)

        return loader, y_mean, y_std

    else:
        return dataset


        # pbar.update(1)
    # pbar.close()
