import lightning as L
import torch
from torch.nn import functional as F

# from torchtune.training import get_cosine_schedule_with_warmup
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from torch_geometric.utils import scatter
from torchmetrics.regression import R2Score

from data.utils import NUM_ATOM_TYPES
from models.layers import (
    CosineCutoff,
    CosineCutoff_2,
    GaussianSmearing,
)
from models.metrics import (
    AUPRC,
    RMSE,
    accuracy,
    count_f1_max,
    pearson,
    recovery,
    spearman,
)
from models.utils import (
    batch_chirality_loss_no_gt,
    batch_chirality_loss_no_gt_before_terminal,
    get_cosine_schedule_with_warmup,
    min_dist_barrier_loss,
    rotation_geodesic_loss,
)

torch.autograd.set_detect_anomaly(True)

from models import layers


def safe_nll_loss(output, y):
    loc = output[:, [0]].clamp(min=-10, max=10)

    # Use softplus instead of exp, and add epsilon
    scale = F.softplus(output[:, 1]) + 0.1  # ensures scale >= 1e-4

    dist = torch.distributions.Normal(loc=loc, scale=scale)
    loss = -dist.log_prob(y).mean()

    # Optional: debug print
    # print(f"μ: {loc.min().item():.2f} ~ {loc.max().item():.2f}, σ: {scale.min().item():.2e} ~ {scale.max().item():.2f}")

    return loss


class AbstractModel(L.LightningModule):
    def __init__(self, cfg, y_mean=None, y_std=None):
        super().__init__()
        self.cfg = cfg
        self.loss_function = cfg.trainer.loss_function
        if cfg.task in [
            "Conformation_generation",
            "Docking",
            "JointDockingConformation",
        ]:
            self.model = COSMolTaskModel.new_docking_conformation(
                cfg.model, cfg.model.backbone, cfg.model.output
            )
        elif cfg.task == "JointPocketPrediction":
            self.model = COSMolTaskModel.new_pocket_pred(
                cfg.model, cfg.model.backbone, cfg.model.output
            )
        elif cfg.task == "Joint_all":
            self.model = COSMolTaskModel.new_joint_all(
                cfg.model, cfg.model.backbone, cfg.model.output
            )
        elif cfg.task == "Docking_pose_ranker":
            self.model = Docking_pose_ranker(
                cfg.model, cfg.model.backbone, cfg.model.output
            )
        elif cfg.task == "PocketPrediction":
            self.model = PocketPrediction(
                cfg.model, cfg.model.backbone, cfg.model.output
            )
        elif cfg.task == "PocketRanker":
            self.model = PocketRanker(cfg.model, cfg.model.backbone, cfg.model.output)

        if y_mean is not None:
            self.y_mean = y_mean.to(self.device)
            self.y_std = y_std.to(self.device)
        else:
            self.y_mean = None
        if self.loss_function == "CrossEntropyLoss":
            print("Using CrossEntropyLoss")
            self.Loss = torch.nn.CrossEntropyLoss()
        elif self.loss_function == "BCEWithLogitsLoss":
            print("Using BCEWithLogitsLoss")
            pos_weight = cfg.trainer.get("pos_weight", None)
            if pos_weight is not None:
                pos_weight = torch.tensor(float(pos_weight))
                print(f"Using BCEWithLogitsLoss pos_weight={pos_weight.item()}")
                self.Loss = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            else:
                self.Loss = torch.nn.BCEWithLogitsLoss()
        elif self.loss_function == "MAELoss":
            # print("Using MAELoss")
            self.Loss = torch.nn.L1Loss()
        elif self.loss_function == "MSELoss":
            # print("Using MSELoss")
            self.Loss = torch.nn.MSELoss()
        elif self.loss_function == "HuberLoss":
            delta = float(cfg.trainer.get("huber_delta", 1.0))
            # print(f"Using HuberLoss delta={delta}")
            self.Loss = torch.nn.HuberLoss(delta=delta)
        elif self.loss_function == "NLL":
            # print("Using NLL")
            self.Loss = safe_nll_loss
        elif self.loss_function == "Displacement":
            # print("Using Displacement")
            self.Loss = lambda output, y: (output - y).abs().mean()
        elif self.loss_function == "RMSD":
            # print("Using RMSD")
            self.Loss = lambda output, y: torch.sqrt(torch.mean((output - y) ** 2))
        elif self.loss_function == "pose_ranker_loss":
            self.Loss = lambda output, y: (
                torch.nn.L1Loss(reduction="none")(output, y) * (y + 0.5)
            ).mean()
        self.MAELoss = torch.nn.L1Loss()

        assert getattr(self, "Loss", None) is not None, "Loss function is not set."

        self.metrics = {
            "f1_max": lambda output, y: count_f1_max(output, y),
            "accuracy": lambda output, y: accuracy(output, y),
            "recovery": lambda output, y: recovery(output, y),
            "MAE": lambda output, y: torch.nn.L1Loss()(output, y),
            "AUPRC": lambda output, y: AUPRC(output, y),
            "RMSE": lambda output, y: RMSE(output, y),
            "Pearson": lambda output, y: pearson(output, y),
            "Spearman": lambda output, y: spearman(output, y),
            "R2": lambda output, y: R2Score()(output, y),
            "Displacement": lambda output, y: (output - y).abs().mean(),
            "RMSD": lambda output, y: torch.sqrt(torch.mean((output - y) ** 2)),
            "MAD": lambda output, y: torch.sqrt(
                torch.sum((output - y) ** 2, dim=-1)
            ).mean(),
        }
        self.metrics = {
            k: v for k, v in self.metrics.items() if k in self.cfg.trainer.metrics
        }
        self.training_step_outputs = []
        self.training_step_y = []
        self.validation_step_outputs = []
        self.validation_step_y = []
        self.test_step_outputs = []
        self.test_step_y = []

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def _metric_for_log(self, value):
        return value.to(self.device) if isinstance(value, torch.Tensor) else value

    def training_step(self, batched_data, batch_idx):
        if self.cfg.task == "JointDockingConformation":
            return self.step_joint_docking_conformation(batched_data)
        if self.cfg.task == "JointPocketPrediction":
            return self.step_joint_pocket_prediction(batched_data, stage="train")
        if self.cfg.task == "Joint_all":
            return self.step_joint_pocket_prediction_docking_conformation(
                batched_data, stage="train"
            )
        if self.cfg.task == "PocketRanker":
            return self.step_property_prediction(batched_data, batch_idx)
        if self.cfg.task == "Conformation_generation":
            return self.step_conformation(batched_data)
        if self.cfg.task == "Docking":
            return self.step_docking(batched_data)
        if self.cfg.task == "PocketPrediction":
            return self.step_pocket_pred(batched_data, batch_idx)
        if self.cfg.task == "Docking_pose_ranker":
            return self.step_docking_pose_ranker(batched_data, batch_idx)

    def step_property_prediction(self, batched_data, batch_idx, stage="train"):
        output = self(batched_data)
        p_pred = output

        if self.y_mean is not None:
            loss = self.Loss(output, (batched_data.y - self.y_mean) / self.y_std)
            p_pred = p_pred * self.y_std + self.y_mean
        else:
            loss = self.Loss(output, batched_data.y)

        if stage == "train":
            self.training_step_outputs.append(p_pred)
            self.training_step_y.append(batched_data.y)
        elif stage == "val":
            self.validation_step_outputs.append(p_pred)
            self.validation_step_y.append(batched_data.y)
        elif stage == "test":
            self.test_step_outputs.append(p_pred)
            self.test_step_y.append(batched_data.y)

        if torch.isnan(loss):
            print(
                p_pred.max(), p_pred.min(), batched_data.y.max(), batched_data.y.min()
            )
            print("Nan detected")
            exit()

        prefix = "" if stage == "train" else "val_" if stage == "val" else "test_"

        self.log(
            prefix + "loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=batched_data.y.shape[0],
        )

        return loss

    def on_train_epoch_end(self):
        if len(self.training_step_outputs) == 0:
            self.log(
                "lr",
                self.scheduler.get_last_lr()[0],
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
                logger=True,
            )
            return

        all_preds = torch.cat(self.training_step_outputs, dim=0)
        all_y = torch.cat(self.training_step_y, dim=0)
        for metric_name, metric in self.metrics.items():
            self.log(
                metric_name,
                self._metric_for_log(metric(all_preds, all_y)),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
                logger=True,
            )
        self.training_step_outputs.clear()
        self.training_step_y.clear()
        self.log(
            "lr",
            self.scheduler.get_last_lr()[0],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            logger=True,
        )

    def test_step(self, batched_data, batch_idx):
        if self.cfg.task == "JointDockingConformation":
            return self.step_joint_docking_conformation(batched_data, stage="test")
        if self.cfg.task == "Conformation_generation":
            return self.step_conformation(batched_data, stage="test")
        if self.cfg.task == "Docking":
            return self.step_docking(batched_data, stage="test")
        if self.cfg.task == "JointPocketPrediction":
            return self.step_joint_pocket_prediction(batched_data, stage="test")
        if self.cfg.task == "Joint_all":
            return self.step_joint_pocket_prediction_docking_conformation(
                batched_data, stage="test"
            )
        if self.cfg.task == "PocketRanker":
            return self.step_property_prediction(batched_data, batch_idx, stage="test")
        output = self(batched_data)

        if (
            self.cfg.task == "Diffusion_GEN"
            or self.cfg.task == "QMugs_opt"
            or self.cfg.task == "Docking"
        ):
            output = output[0]

        if self.cfg.task == "Docking_pose_ranker":
            return self.step_docking_pose_ranker(batched_data, stage="test")

        if self.cfg.task == "PocketPrediction":
            output = output[1]

        if self.cfg.task == "PocketPredictionScore":
            output = output[0]

        if self.loss_function == "NLL":
            p_pred = output[:, [0]]
        else:
            p_pred = output

        if self.y_mean != None:
            loss = self.Loss(output, (batched_data.y - self.y_mean) / self.y_std)
            p_pred = p_pred * self.y_std + self.y_mean
        else:
            loss = self.Loss(output, batched_data.y)

        self.test_step_outputs.append(p_pred)
        self.test_step_y.append(batched_data.y)

        self.log(
            "test_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=batched_data.y.shape[0],
        )

    def on_test_epoch_end(self):
        if len(self.test_step_outputs) == 0:
            return

        all_preds = torch.cat(self.test_step_outputs, dim=0)
        all_y = torch.cat(self.test_step_y, dim=0)
        for metric_name, metric in self.metrics.items():
            self.log(
                "test_" + metric_name,
                self._metric_for_log(metric(all_preds, all_y)),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
            )
        self.test_step_outputs.clear()
        self.test_step_y.clear()

    def validation_step(self, batched_data, batch_idx):
        if self.cfg.task == "JointDockingConformation":
            return self.step_joint_docking_conformation(batched_data, stage="val")
        if self.cfg.task == "Conformation_generation":
            return self.step_conformation(batched_data, stage="val")
        if self.cfg.task == "JointPocketPrediction":
            return self.step_joint_pocket_prediction(batched_data, stage="val")
        if self.cfg.task == "Joint_all":
            return self.step_joint_pocket_prediction_docking_conformation(
                batched_data, stage="val"
            )
        if self.cfg.task == "PocketRanker":
            return self.step_property_prediction(batched_data, batch_idx, stage="val")
        elif self.cfg.task == "Docking_pose_ranker":
            return self.step_docking_pose_ranker(batched_data, stage="val")
        elif self.cfg.task == "Docking":
            return self.step_docking(batched_data, stage="val")
        elif self.cfg.task == "PocketPrediction":
            return self.step_pocket_pred(batched_data, batch_idx, stage="val")

        output = self(batched_data)

        if self.loss_function == "NLL":
            p_pred = output[:, [0]]
        else:
            p_pred = output

        if self.y_mean != None:
            loss = self.Loss(output, (batched_data.y - self.y_mean) / self.y_std)
            p_pred = p_pred * self.y_std + self.y_mean
        else:
            loss = self.Loss(output, batched_data.y)

        self.validation_step_outputs.append(p_pred)
        self.validation_step_y.append(batched_data.y)

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=batched_data.y.shape[0],
        )

    def on_validation_epoch_end(self):
        if len(self.validation_step_outputs) == 0:
            return

        all_preds = torch.cat(self.validation_step_outputs, dim=0)
        all_y = torch.cat(self.validation_step_y, dim=0)
        for metric_name, metric in self.metrics.items():
            self.log(
                "val_" + metric_name,
                self._metric_for_log(metric(all_preds, all_y)),
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
                logger=True,
            )
        self.validation_step_outputs.clear()
        self.validation_step_y.clear()


    def step_conformation(self, batched_data, stage="train", prefix="conf"):
        # batched_data = batched_data["conf"]
        if getattr(batched_data, "ligand_mask", None) is None:
            batched_data.ligand_mask = torch.ones(
                batched_data.x.shape[0],
                dtype=torch.bool,
                device=batched_data.x.device,
            )
        v_t = self(batched_data, "conformation")

        t = batched_data.t
        pos = batched_data.pos

        x0_hat = pos + v_t

        if stage == "train":
            self.training_step_outputs.append(x0_hat.detach().cpu())
            self.training_step_y.append(batched_data.y.detach().cpu())
        elif stage == "val":
            self.validation_step_outputs.append(x0_hat.detach().cpu())
            self.validation_step_y.append(batched_data.y.detach().cpu())
        elif stage == "test":
            self.test_step_outputs.append(x0_hat.detach().cpu())
            self.test_step_y.append(batched_data.y.detach().cpu())

        loss_factor = (
            (1 - t)
            .abs()
            .clamp_min(self.cfg.trainer.get("min_loss_factor", 0.03))
            .unsqueeze(1)
        )
        mse_loss = self.Loss(v_t / loss_factor, batched_data.u_t / loss_factor)

        chirality_loss = batch_chirality_loss_no_gt_before_terminal(
            x0_hat,
            batched_data.chiral_neighbors,
            batched_data.chirality,
            t,
            terminal=0.8,
        )

        if torch.isnan(mse_loss):
            print("Nan detected")
            exit()

        loss = mse_loss + chirality_loss

        prefix = f"{prefix}_" if stage == "train" else f"{stage}_{prefix}_"

        self.log(
            prefix + "mse_loss",
            mse_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=batched_data.y.shape[0],
        )

        self.log(
            prefix + "chi_loss",
            chirality_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=batched_data.y.shape[0],
        )

        if loss > 30.0 or torch.isnan(loss):
            # print("Loss:", loss, chirality_loss)
            # print("Loss:", loss, chirality_loss)
            # print(
            #     eps_theta.max(),
            #     eps_theta.min(),
            #     batched_data.y.max(),
            #     batched_data.y.min(),
            # )
            # loss = torch.log10(loss - 20) * 10 + 20
            # loss = torch.clamp(loss, 0, 10)
            # exit()
            0

        self.log(
            prefix + "loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=batched_data.y.shape[0],
        )

        return loss

    def step_joint_docking_conformation(self, batch, stage="train"):
        if isinstance(batch, tuple):
            batch = batch[0]

        conf_batch = batch["conf"]
        dock_batch = batch["dock"]
        conf_loss = self.step_conformation(conf_batch, stage=stage)
        dock_loss = self.step_docking(dock_batch, stage=stage)
        loss = conf_loss * self.cfg.trainer.get(
            "conf_loss_weight", 1.0
        ) + dock_loss * self.cfg.trainer.get("dock_loss_weight", 1.0)

        prefix = "" if stage == "train" else f"{stage}_"
        batch_size = getattr(conf_batch, "num_graphs", 0) + getattr(
            dock_batch, "num_graphs", 0
        )
        self.log(
            prefix + "loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=batch_size,
        )

        return loss

    def step_docking(self, batched_data, stage="train", prefix="dock", task_id=None):
        if isinstance(batched_data, dict):
            batched_data = batched_data["dock"]
        v_t, R_pred, shift = self(batched_data, "docking")

        t = batched_data.t[batched_data.ligand_mask]
        pos_ligand = batched_data.pos[batched_data.ligand_mask]
        t_batch = scatter(
            t, batched_data.batch[batched_data.ligand_mask], dim=0, reduce="mean"
        )

        x0_hat = pos_ligand + v_t
        batched_data.pos[batched_data.ligand_mask] = x0_hat

        if stage == "train":
            self.training_step_outputs.append(x0_hat.detach().cpu())
            self.training_step_y.append(batched_data.y.detach().cpu())
        elif stage == "val":
            self.validation_step_outputs.append(x0_hat.detach().cpu())
            self.validation_step_y.append(batched_data.y.detach().cpu())
        elif stage == "test":
            self.test_step_outputs.append(x0_hat.detach().cpu())
            self.test_step_y.append(batched_data.y.detach().cpu())

        loss_factor = (
            (1 - t)
            .abs()
            .clamp_min(self.cfg.trainer.get("min_loss_factor", 0.03))
            .unsqueeze(1)
        )

        mse_loss = self.MAELoss(
            v_t / loss_factor, batched_data.u_t / loss_factor
        )  # 硬编码需要优化
        rotation_loss = rotation_geodesic_loss(R_pred, batched_data.R_pred, 1)
        shift_loss = self.MAELoss(shift, batched_data.shift)
        chirality_loss = batch_chirality_loss_no_gt(
            x0_hat, batched_data.chiral_neighbors_ligand, batched_data.chirality_ligand
        )
        mindist_loss = min_dist_barrier_loss(
            pos_pred=batched_data.pos,
            edge_index=batched_data.edge_index,
            mindist=batched_data.edge_mindist,
            t_graph=t_batch,
            batch=batched_data.batch,
            t_threshold=0.85,
        )

        if torch.isnan(mse_loss):
            print("Nan detected")
            exit()

        loss = (
            mse_loss
            + rotation_loss * 0.15
            + shift_loss * 0.15
            + chirality_loss * 0.6
            + mindist_loss * 2e2
        )

        stage_prefix = "" if stage == "train" else "val_" if stage == "val" else "test_"
        prefix = stage_prefix + (f"{prefix}_" if prefix else "")

        self.log(
            prefix + "mse_loss",
            mse_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=batched_data.y.shape[0],
        )

        self.log(
            prefix + "rotation_loss",
            rotation_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=batched_data.y.shape[0],
        )

        self.log(
            prefix + "shift_loss",
            shift_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=batched_data.y.shape[0],
        )

        self.log(
            prefix + "loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=batched_data.y.shape[0],
        )

        self.log(
            prefix + "chi_loss",
            chirality_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=batched_data.y.shape[0],
        )

        self.log(
            prefix + "mindist_loss",
            mindist_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=batched_data.y.shape[0],
        )

        return loss

    def step_docking_pose_ranker(self, batched_data, stage="train"):
        batched_data = batched_data.extract_pocket_and_update_graph(
            cutoff=self.model.radius_cutoff,
            max_num_neighbors=999,
        )
        score = self(batched_data)

        score = score.squeeze(-1)

        if stage == "train":
            self.training_step_outputs.append(score)
            self.training_step_y.append(batched_data.y)
        elif stage == "val":
            self.validation_step_outputs.append(score)
            self.validation_step_y.append(batched_data.y)
        elif stage == "test":
            self.test_step_outputs.append(score)
            self.test_step_y.append(batched_data.y)

        loss = self.Loss(score, batched_data.y)

        if torch.isnan(loss):
            print("Nan detected")
            exit()

        prefix = "" if stage == "train" else "val_" if stage == "val" else "test_"

        self.log(
            prefix + "loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=batched_data.y.shape[0],
        )

        return loss

    def step_joint_pocket_prediction(self, batched_data, stage="train"):
        if isinstance(batched_data, tuple):
            batched_data = batched_data[0]
        pocket_data = batched_data["pocket"]
        pocket_ranker_data = batched_data["pocket_ranker"]

        pocket_vec_loss = self.step_pocket_pred(pocket_data, stage=stage)
        pocket_ranker_loss = self.step_pocket_ranker(pocket_ranker_data, stage=stage)
        loss = pocket_vec_loss * self.cfg.trainer.get(
            "pocket_vec_loss_weight", 1.0
        ) + pocket_ranker_loss * self.cfg.trainer.get("pocket_ranker_loss_weight", 1.0)

        prefix = "" if stage == "train" else f"{stage}_"
        batch_size = getattr(pocket_data, "num_graphs", 0) + getattr(
            pocket_ranker_data, "num_graphs", 0
        )
        self.log(
            prefix + "loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=batch_size,
        )

        return loss

    def step_joint_pocket_prediction_docking_conformation(
        self, batched_data, stage="train"
    ):

        if isinstance(batched_data, tuple):
            batched_data = batched_data[0]
        pocket_data = batched_data["pocket"]
        pocket_ranker_data = batched_data["pocket_ranker"]
        conf_batch = batched_data["conf"]
        dock_batch = batched_data["dock"]

        conf_loss = self.step_conformation(conf_batch, stage=stage)
        dock_loss = self.step_docking(dock_batch, stage=stage)
        pocket_vec_loss = self.step_pocket_pred(pocket_data, stage=stage)
        pocket_ranker_loss = self.step_pocket_ranker(pocket_ranker_data, stage=stage)
        loss = (
            conf_loss * self.cfg.trainer.get("conf_loss_weight", 1.0)
            + dock_loss * self.cfg.trainer.get("dock_loss_weight", 1.0)
            + pocket_vec_loss * self.cfg.trainer.get("pocket_vec_loss_weight", 1.0)
            + pocket_ranker_loss
            * self.cfg.trainer.get("pocket_ranker_loss_weight", 1.0)
        )

        prefix = "" if stage == "train" else f"{stage}_"
        batch_size = getattr(conf_batch, "num_graphs", 0) + getattr(
            dock_batch, "num_graphs", 0
        )
        self.log(
            prefix + "loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=batch_size,
        )

        return loss

    def step_pocket_pred(self, batched_data, batch_idx=None, stage="train", prefix=""):
        if "pocket_pred" in getattr(self.model, "available_tasks", set()):
            vec_to_pocket = self(batched_data, "pocket_pred")
            from models.pocket_losses import pocket_vector_l1_loss

            proximity_weights = (
                batched_data.y
                if self.cfg.trainer.get("pocket_proximity_weighting", False)
                else None
            )
            pocket_vec_loss = pocket_vector_l1_loss(
                vec_to_pocket, batched_data.pocket_vec, proximity_weights
            )

            prefix = (
                f"{prefix}_"
                if prefix
                else ""
                if stage == "train"
                else "val_"
                if stage == "val"
                else "test_"
            )
            self.log(
                prefix + "pocket_vec_loss",
                pocket_vec_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
                batch_size=batched_data.y.shape[0],
            )

            return pocket_vec_loss

        vec_to_pocket, score = self(batched_data)

        if stage == "train":
            self.training_step_outputs.append(score)
            self.training_step_y.append(batched_data.y)
        elif stage == "val":
            self.validation_step_outputs.append(score)
            self.validation_step_y.append(batched_data.y)
        elif stage == "test":
            self.test_step_outputs.append(score)
            self.test_step_y.append(batched_data.y)

        mse_loss = self.Loss(score, batched_data.y)

        pocket_loss = (
            (vec_to_pocket - batched_data.pocket_vec).norm(dim=-1) * batched_data.y
        ).mean()

        if torch.isnan(mse_loss):
            print("Nan detected")
            exit()

        loss = mse_loss * 10.0 + pocket_loss

        prefix = "" if stage == "train" else "val_" if stage == "val" else "test_"

        self.log(
            prefix + "mse_loss",
            mse_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=batched_data.y.shape[0],
        )

        self.log(
            prefix + "pocket_loss",
            pocket_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=batched_data.y.shape[0],
        )

        self.log(
            prefix + "loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=batched_data.y.shape[0],
        )

        return loss

    def step_pocket_ranker(
        self, batched_data, batch_idx=None, stage="train", prefix=""
    ):
        if "pocket_ranker" in getattr(self.model, "available_tasks", set()):
            score = self(batched_data, "pocket_ranker")
            loss = torch.nn.L1Loss()(score, batched_data.y)

            prefix = (
                f"{prefix}_"
                if prefix
                else ""
                if stage == "train"
                else "val_"
                if stage == "val"
                else "test_"
            )
            self.log(
                prefix + "pocket_ranker_loss",
                loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
                sync_dist=True,
                batch_size=batched_data.y.shape[0],
            )

            return loss

        score = self(batched_data)

        if stage == "train":
            self.training_step_outputs.append(score)
            self.training_step_y.append(batched_data.y)
        elif stage == "val":
            self.validation_step_outputs.append(score)
            self.validation_step_y.append(batched_data.y)
        elif stage == "test":
            self.test_step_outputs.append(score)
            self.test_step_y.append(batched_data.y)

        mse_loss = self.Loss(score, batched_data.y)

        if torch.isnan(mse_loss):
            print("Nan detected")
            exit()

        loss = mse_loss

        prefix = "" if stage == "train" else "val_" if stage == "val" else "test_"

        self.log(
            prefix + "mse_loss",
            mse_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=batched_data.y.shape[0],
        )

        self.log(
            prefix + "loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=batched_data.y.shape[0],
        )

        return loss

    def configure_optimizers(self):
        opt_name = self.cfg.trainer.optimizer["class"]
        if opt_name == "AdamW":
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.cfg.trainer.optimizer.lr,
                weight_decay=self.cfg.trainer.optimizer.weight_decay,
            )
        if opt_name == "Adam":
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.cfg.trainer.optimizer.lr,
                weight_decay=self.cfg.trainer.optimizer.weight_decay,
            )
        elif opt_name == "SGD":
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.cfg.trainer.optimizer.lr,
                weight_decay=self.cfg.trainer.optimizer.weight_decay,
                momentum=self.cfg.trainer.optimizer.momentum,
            )

        scheduler = self.cfg.trainer.scheduler.get("class", None)

        if scheduler == "ReduceLROnPlateau":
            self.scheduler = ReduceLROnPlateau(
                optimizer,
                mode=self.cfg.trainer.scheduler.mode,
                factor=self.cfg.trainer.scheduler.factor,
                patience=self.cfg.trainer.scheduler.patience,
            )
        elif scheduler == "StepLR":
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.cfg.trainer.scheduler.step_size,
                gamma=self.cfg.trainer.scheduler.gamma,
            )
        elif scheduler == "CosineAnnealingWarmRestarts":
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer,
                T_0=self.cfg.trainer.scheduler.T_0,
                T_mult=self.cfg.trainer.scheduler.T_mult,
                eta_min=self.cfg.scheduler.eta_min,
            )
        elif scheduler == "MultiStepLR":
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=self.cfg.trainer.scheduler.milestones,
                gamma=self.cfg.trainer.scheduler.gamma,
            )
        elif scheduler == "ExponentialLR":
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=self.cfg.trainer.scheduler.gamma
            )
        elif scheduler == "CosineWithWarmup":
            self.scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.cfg.trainer.scheduler.num_warmup_steps,
                num_training_steps=self.cfg.trainer.scheduler.num_training_steps,
                num_cycles=self.cfg.trainer.scheduler.num_cycles,
            )
        else:
            self.scheduler = LambdaLR(optimizer, lr_lambda=lambda step: 1.0)

        lr_scheduler_config = {
            "scheduler": self.scheduler,
            "monitor": self.cfg.trainer.scheduler.get("monitor", None),
            "interval": "epoch",
        }

        return [optimizer], [lr_scheduler_config]


class Docking_pose_ranker(torch.nn.Module):
    def __init__(self, cfg_model, cfg_backbone, cfg_output):
        super().__init__()
        self.radius_cutoff = cfg_backbone.radius_cutoff
        self.atom_encoder = AtomEncoder(cfg_model.input_dim, use_ligand_mask=True)
        self.edge_encoder = EdgeEncoder(cfg_model.input_dim)
        self.representation = COSMol(cfg_model.input_dim, cfg_backbone)
        self.output = getattr(layers, cfg_output.name)(cfg_model.input_dim, cfg_output)

    def forward(self, batched_data):
        required_fields = ("edge_index", "edge_attr", "edge_vector")
        missing_fields = [
            field
            for field in required_fields
            if not hasattr(batched_data, field) or getattr(batched_data, field) is None
        ]
        if missing_fields:
            raise ValueError(
                "Docking_pose_ranker requires a prepared graph; missing fields: "
                + ", ".join(missing_fields)
            )

        node_feats = self.atom_encoder(
            batched_data.x, None, None, batched_data.ligand_mask
        )
        edge_feats = self.edge_encoder(batched_data.edge_attr)

        binding_representation = self.representation(
            node_feats,
            edge_feats,
            batched_data.edge_index,
            batched_data.edge_vector,
            batched_data.batch,
        )

        return self.output(binding_representation, batched_data.ligand_mask)


class PocketPrediction(torch.nn.Module):
    def __init__(self, cfg_model, cfg_backbone, cfg_output):
        super().__init__()
        self.atom_type = torch.nn.Embedding(5, cfg_model.input_dim)
        self.residu_type = torch.nn.Embedding(25, cfg_model.input_dim)
        self.edge_type = torch.nn.Embedding(30, cfg_model.input_dim)

        self.representation = COSMol(cfg_model.input_dim, cfg_backbone)
        self.output = getattr(layers, cfg_output.name)(cfg_model.input_dim, cfg_output)

    def forward(self, batched_data):
        node_feats = self.atom_type(batched_data.node_types_atom) + self.residu_type(
            batched_data.node_types_residue
        )
        edge_feats = self.edge_type(batched_data.edge_types)

        protein_representation = self.representation(
            node_feats,
            edge_feats,
            batched_data.edge_index,
            batched_data.edge_vector,
            batched_data.batch,
        )

        return self.output(protein_representation, batched_data.CB_mask)


class COSMolTaskModel(torch.nn.Module):
    def __init__(self, cfg_model, cfg_backbone, cfg_output, init_mode):
        super().__init__()
        self.available_tasks = set()
        self.representation = COSMol(cfg_model.input_dim, cfg_backbone)

        if init_mode == "docking_conformation":
            self.init_docking_conformation(cfg_model, cfg_backbone, cfg_output)
        elif init_mode == "pocket_pred":
            self.init_pocket_pred(cfg_model, cfg_backbone, cfg_output)
            self.init_pocket_ranker(cfg_model, cfg_backbone, cfg_output)
        elif init_mode == "joint_all":
            self.init_docking_conformation(cfg_model, cfg_backbone, cfg_output)
            self.init_pocket_pred(cfg_model, cfg_backbone, cfg_output)
            self.init_pocket_ranker(cfg_model, cfg_backbone, cfg_output)
        else:
            raise ValueError(f"Unknown init_mode: {init_mode}")

    @classmethod
    def new_docking_conformation(cls, cfg_model, cfg_backbone, cfg_output):
        return cls(cfg_model, cfg_backbone, cfg_output, "docking_conformation")

    @classmethod
    def new_pocket_pred(cls, cfg_model, cfg_backbone, cfg_output):
        return cls(cfg_model, cfg_backbone, cfg_output, "pocket_pred")

    @classmethod
    def new_joint_all(cls, cfg_model, cfg_backbone, cfg_output):
        return cls(cfg_model, cfg_backbone, cfg_output, "joint_all")

    def init_docking_conformation(self, cfg_model, cfg_backbone, cfg_output):
        self.terminal_time_threshold = cfg_model.get("terminal_time_threshold", None)
        self.atom_encoder = AtomEncoder(
            cfg_model.input_dim,
            use_chirality_feat=True,
            use_ligand_mask=True,
            use_t=True,
            use_task_feat=True,
            t_rbf_dim=cfg_model.t_rbf_dim,
        )
        self.edge_encoder = EdgeEncoder(cfg_model.input_dim, use_chirality_feat=True)
        self.conf_num_layers = cfg_backbone.get("conf_num_layers", None)
        self.dock_num_layers = cfg_backbone.get("dock_num_layers", None)
        if cfg_output.get("dock", None) is not None:
            self.available_tasks.add("docking")
            self.dock_output = getattr(layers, cfg_output.dock.name)(
                cfg_model.input_dim, cfg_output.dock
            )
        if cfg_output.get("conf", None) is not None:
            self.available_tasks.add("conformation")
            self.conf_output = getattr(layers, cfg_output.conf.name)(
                cfg_model.input_dim, cfg_output.conf
            )

    def init_pocket_pred(self, cfg_model, cfg_backbone, cfg_output):
        self.available_tasks.add("pocket_pred")
        self.pocket_atom_type = torch.nn.Embedding(5, cfg_model.input_dim)
        self.pocket_residue_type = torch.nn.Embedding(25, cfg_model.input_dim)
        self.pocket_edge_type = torch.nn.Embedding(30, cfg_model.input_dim)
        self.pocket_num_layers = cfg_backbone.get("pocket_num_layers", None)
        self.pocket_output = getattr(layers, cfg_output.pocket.name)(
            cfg_model.input_dim, cfg_output.pocket
        )

    def init_pocket_ranker(self, cfg_model, cfg_backbone, cfg_output):
        self.available_tasks.add("pocket_ranker")
        self.pocket_ranker_atom_encoder = AtomEncoder(
            cfg_model.input_dim, num_atom_types=NUM_ATOM_TYPES + 1
        )
        self.pocket_ranker_edge_type = torch.nn.Embedding(1, cfg_model.input_dim)
        self.pocket_ranker_num_layers = cfg_backbone.get(
            "pocket_ranker_num_layers", None
        )
        pocket_ranker_output_cfg = cfg_output.get("pocket_ranker", None)
        if pocket_ranker_output_cfg is None:
            raise ValueError("Missing model.output.pocket_ranker")
        self.pocket_ranker_output = getattr(layers, pocket_ranker_output_cfg.name)(
            cfg_model.input_dim, pocket_ranker_output_cfg
        )

    def _model_time(self, t):
        if self.terminal_time_threshold is None:
            return t
        return torch.where(
            t > self.terminal_time_threshold,
            torch.ones_like(t),
            t,
        )

    def forward_docking_conformation(self, batched_data, task):
        t_model = self._model_time(batched_data.t)
        node_feats = self.atom_encoder(
            batched_data.x,
            t_model,
            batched_data.chirality,
            batched_data.ligand_mask,
            1 if task == "conformation" else 0,
        )
        edge_feats = self.edge_encoder(
            batched_data.edge_attr, batched_data.edge_chirality_feat
        )

        num_layers = (
            self.conf_num_layers if task == "conformation" else self.dock_num_layers
        )
        representation = self.representation(
            node_feats,
            edge_feats,
            batched_data.edge_index,
            batched_data.edge_vector,
            batched_data.batch,
            num_layers=num_layers,
        )

        if task == "conformation":
            return self.conf_output(representation, batched_data.pos, t_model)

        return self.dock_output(
            representation,
            batched_data.pos,
            t_model,
            batched_data.ligand_mask,
        )

    def forward_pocket_pred(self, batched_data):
        node_feats = self.pocket_atom_type(
            batched_data.node_types_atom
        ) + self.pocket_residue_type(batched_data.node_types_residue)
        edge_feats = self.pocket_edge_type(batched_data.edge_types)

        protein_representation = self.representation(
            node_feats,
            edge_feats,
            batched_data.edge_index,
            batched_data.edge_vector,
            batched_data.batch,
            num_layers=self.pocket_num_layers,
        )

        return self.pocket_output(protein_representation, batched_data.CB_mask)

    def forward_pocket_ranker(self, batched_data):
        node_feats = self.pocket_ranker_atom_encoder(batched_data.x)
        edge_feats = self.pocket_ranker_edge_type(batched_data.edge_types)

        protein_representation = self.representation(
            node_feats,
            edge_feats,
            batched_data.edge_index,
            batched_data.edge_vector,
            batched_data.batch,
            num_layers=self.pocket_ranker_num_layers,
        )

        return self.pocket_ranker_output(protein_representation, batched_data.x)

    def forward(self, batched_data, task):
        if task not in self.available_tasks:
            raise ValueError(
                f"Task {task} was not initialized. "
                f"Available tasks: {sorted(self.available_tasks)}"
            )
        if task in ["conformation", "docking"]:
            return self.forward_docking_conformation(batched_data, task)
        if task == "pocket_pred":
            return self.forward_pocket_pred(batched_data)
        if task == "pocket_ranker":
            return self.forward_pocket_ranker(batched_data)
        raise ValueError(f"Unknown task: {task}")


class PocketRanker(torch.nn.Module):
    def __init__(self, cfg_model, cfg_backbone, cfg_output):
        super().__init__()
        self.atom_encoder = AtomEncoder(
            cfg_model.input_dim, num_atom_types=NUM_ATOM_TYPES + 1
        )
        self.edge_type = torch.nn.Embedding(1, cfg_model.input_dim)
        self.representation = COSMol(cfg_model.input_dim, cfg_backbone)
        self.output = getattr(layers, cfg_output.name)(cfg_model.input_dim, cfg_output)

    def forward(self, batched_data):
        node_feats = self.atom_encoder(batched_data.x)
        edge_feats = self.edge_type(batched_data.edge_types)

        protein_representation = self.representation(
            node_feats,
            edge_feats,
            batched_data.edge_index,
            batched_data.edge_vector,
            batched_data.batch,
        )

        return self.output(protein_representation, batched_data.x)


class COSMol(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        cfg,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_layers = cfg.num_layers
        self.dist_rbf_dim = cfg.dist_rbf_dim
        self.radius_cutoff = cfg.radius_cutoff
        assert cfg.get("normalization", False) in [False, "preDyt", "postDyt"], (
            f"normalization must be one of false, 'preDyt', 'postDyt' instead of {cfg.normalization}"
        )

        self.rbf = GaussianSmearing(
            0.0, self.radius_cutoff, self.dist_rbf_dim, trainable=True
        )
        self.rbf_liner = torch.nn.Linear(self.dist_rbf_dim, self.input_dim)

        self.interactions = torch.nn.ModuleList(
            [
                layers.COSMol_interaction(input_dim=self.input_dim, cfg=cfg)
                for _ in range(self.num_layers)
            ]
        )

        self.update = torch.nn.ModuleList(
            [
                layers.COSMol_update(self.input_dim, cfg=cfg)
                for _ in range(self.num_layers)
            ]
        )

        self.deep_residual_span = cfg.get("deep_residual_span", 0)
        if type(self.deep_residual_span) is not int or not 0 <= self.deep_residual_span <= self.num_layers:
            raise ValueError("deep_residual_span must be an integer between 0 and num_layers")
        if self.deep_residual_span:
            # Index s controls the edge s -> s + span; node 0 is the input.
            self.deep_residual_strengths = torch.nn.Parameter(
                torch.zeros(self.num_layers - self.deep_residual_span + 1)
            )

        if cfg.cutoff_fn == "Cosine":
            self.cutoff_fn = CosineCutoff(self.radius_cutoff)
        elif cfg.cutoff_fn == "Cosine_2":
            self.cutoff_fn = CosineCutoff_2(self.radius_cutoff)
        elif cfg.cutoff_fn == "IEConvCutoff":
            self.cutoff_fn = lambda d_ij: (
                0.5 - torch.tanh(d_ij / self.radius_cutoff * 16.0 - 14.0) * 0.5
            )
        else:
            self.cutoff_fn = lambda d_ij: torch.ones_like(d_ij)

    def forward(
        self,
        x,
        edge_feats,
        edge_index,
        edge_vector,
        batch,
        num_layers=None,
    ):
        # compute node and pair features
        edge_len = torch.norm(edge_vector, dim=1, keepdim=True) + 1e-3
        edge_vector_n = edge_vector / edge_len
        fcut = self.cutoff_fn(edge_len) if self.cutoff_fn else None
        edge_feats = edge_feats + self.rbf_liner(self.rbf(edge_len))
        x_vec = torch.zeros((x.shape[0], 3, self.input_dim), device=x.device)

        if num_layers is None:
            num_layers = self.num_layers
        num_layers = min(num_layers, self.num_layers)

        history = [(x, x_vec)] if self.deep_residual_span else None
        for i in range(num_layers):
            interaction = self.interactions[i]
            update = self.update[i]
            x, x_vec = interaction(
                x,
                x_vec,
                edge_feats,
                fcut,
                edge_vector_n,
                edge_index,
            )
            x, x_vec = update(x, x_vec)
            if history is not None:
                # Tap the layer output before mixing its incoming deep residual.
                history.append((x, x_vec))
                source = i + 1 - self.deep_residual_span
                if source >= 0:
                    strength = self.deep_residual_strengths[source]
                    skip_x, skip_vec = history[source]
                    x = (1 - strength) * x + strength * skip_x
                    x_vec = (1 - strength) * x_vec + strength * skip_vec
        return (x, x_vec, batch)


class AtomEncoder(torch.nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        use_chirality_feat: bool = False,
        use_ligand_mask: bool = False,
        use_t: bool = False,
        use_task_feat: bool = False,
        t_rbf_dim: int = 0,
        num_atom_types: int = NUM_ATOM_TYPES,
    ):
        super().__init__()
        self.t_rbf_dim = t_rbf_dim
        self.hidden_dim = hidden_dim
        self.use_chirality_feat = use_chirality_feat
        self.use_ligand_mask = use_ligand_mask
        self.use_t = use_t
        self.use_task_feat = use_task_feat
        self.num_atom_types = num_atom_types

        # x: [N, 9]
        # order:
        # 0 atomic_num              119
        # 1 chirality               9
        # 2 degree                  11
        # 3 formal_charge           13
        # 4 num_hs                  9
        # 5 num_radical_electrons   5
        # 6 hybridization           9
        # 7 is_aromatic             2
        # 8 is_in_ring              2

        self.atomic_num_emb = torch.nn.Embedding(self.num_atom_types, hidden_dim)
        # self.chiral_tag_emb = torch.nn.Embedding(9, hidden_dim)
        self.degree_emb = torch.nn.Embedding(11, hidden_dim)
        self.formal_charge_emb = torch.nn.Embedding(13, hidden_dim)
        self.num_hs_emb = torch.nn.Embedding(9, hidden_dim)
        self.num_radical_e_emb = torch.nn.Embedding(5, hidden_dim)
        self.hybridization_emb = torch.nn.Embedding(9, hidden_dim)
        self.is_aromatic_emb = torch.nn.Embedding(2, hidden_dim)
        self.is_in_ring_emb = torch.nn.Embedding(2, hidden_dim)

        if self.use_chirality_feat:
            self.chirality_feat_emb = torch.nn.Embedding(2, hidden_dim)
        if self.use_ligand_mask:
            self.ligand_mask_emb = torch.nn.Embedding(2, hidden_dim)
        if self.use_task_feat:
            self.task_feat_emb = torch.nn.Embedding(2, hidden_dim)
        if self.use_t:
            self.t_rbf = GaussianSmearing(0.0, 1.0, self.t_rbf_dim, trainable=True)
            self.n_step_liner = torch.nn.Linear(self.t_rbf_dim, hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor = None,
        chirality: torch.Tensor = None,
        ligand_mask: torch.Tensor = None,
        task_id: int = None,
    ) -> torch.Tensor:
        assert x.dtype == torch.long
        # assert x.size(-1) == 2, f"Expected x shape [N, 2], got {x.shape}" # tmp
        # assert x.size(-1) == 1, f"Expected x shape [N, 1], got {x.shape}"

        h = self.atomic_num_emb(x[:, 0])
        if self.use_t:
            h = h + self.n_step_liner(self.t_rbf(t.unsqueeze(-1)))
        if self.use_chirality_feat:
            h = h + self.chirality_feat_emb(chirality)
        if self.use_ligand_mask:
            h = h + self.ligand_mask_emb(ligand_mask.long())
        if self.use_task_feat:
            h = h + self.task_feat_emb(torch.tensor(task_id, device=x.device))

        return h


class EdgeEncoder(torch.nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        use_chirality_feat: bool = False,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.use_chirality_feat = use_chirality_feat

        # edge_attr: [E, 3]
        # 0 bond_type      23  (includes RADIUS=22)
        # 1 stereo         6
        # 2 is_conjugated  2
        self.bond_type_emb = torch.nn.Embedding(23, hidden_dim)
        self.stereo_emb = torch.nn.Embedding(6, hidden_dim)
        self.is_conjugated_emb = torch.nn.Embedding(2, hidden_dim)

        if use_chirality_feat:
            self.edge_chirality_feat_emb = torch.nn.Embedding(4, hidden_dim)

    def forward(
        self,
        edge_attr: torch.Tensor,
        edge_chirality_feat: torch.Tensor = None,
    ) -> torch.Tensor:
        assert edge_attr.dtype == torch.long
        assert edge_attr.size(-1) == 1, (
            f"Expected edge_attr shape [E, 1], got {edge_attr.shape}"
        )  # tmp
        # assert edge_attr.size(-1) == 1, f"Expected edge_attr shape [E, 1], got {edge_attr.shape}"

        h = (
            self.bond_type_emb(edge_attr[:, 0])
            # + self.stereo_emb(edge_attr[:, 1])
            # + self.is_conjugated_emb(edge_attr[:, 2])
        )

        if self.use_chirality_feat:
            assert edge_chirality_feat is not None
            h = h + self.edge_chirality_feat_emb(edge_chirality_feat)

        return h
