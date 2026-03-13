import torch
import torch.nn as nn
import numpy as np
import config as cfg
import copy
import warnings
from os.path import join
from tqdm import tqdm
from PriorNet import PriorNet 
from D_net import Discriminator
from UALNet import UALNet
from init import *
from einops import rearrange


warnings.filterwarnings("ignore")


def psnr(z, z_hat):
    MAX_VALUE = np.max(z**2, axis=(1,2))+1e-12
    L = z_hat.shape[1] * z_hat.shape[2]
    error = np.sum((z_hat - z)**2, axis=(1,2))/L
    index = 10*np.log10(MAX_VALUE/error)
    m_idx = np.mean(index)

    return m_idx

def sam(z, z_hat):
    de_z_hat = np.sqrt(np.sum(z_hat**2, axis=0))
    de_z = np.sqrt(np.sum(z**2, axis=0))
    angle = np.rad2deg(np.arccos(np.sum(z_hat * z, axis=0)/(de_z_hat * de_z)))
    sam_idx = np.mean(angle, axis=(0,1))

    return sam_idx

def rmse(z, z_hat):
    L = z_hat.shape[1] * z_hat.shape[2]
    M = z_hat.shape[0]
    rmse_m = np.sqrt(np.sum((z_hat - z)**2, axis=(1, 2))) / np.sqrt(L)
    rmse = np.sqrt(np.sum(rmse_m**2)/M)

    return rmse


class Trainer(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        # --------------------------------------------------------------
        # Device and filesystem initialization
        # --------------------------------------------------------------
        self.device = (
            torch.device(f"cuda:{cfg.gpu_ids}")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        self.iter_loss_log_path = join(cfg.save_path, "msg_files_everyloss")
        self.epoch_log_path = join(cfg.save_path, "msg_files")
       

        print("=" * 60)
        init_save(cfg.f_to_save, cfg.save_path, cfg.save_result)
        print("Copying training code and creating experiment folders")
        print("=" * 60)

        # --------------------------------------------------------------
        # Dataloaders
        # --------------------------------------------------------------
        print("=" * 60)
        print("Initializing training, validation, and test dataloaders")
        self.train_loader, self.val_loader, self.test_loader = init_data(
            cfg.train_data_path,
            cfg.test_data_path,
            cfg.valid_data_path,
            cfg.batch_size,
            cfg.patch_size,
        )
        print("=" * 60)

        # --------------------------------------------------------------
        # PriorNet
        # --------------------------------------------------------------
        print("=" * 60)
        print("Initializing PriorNet")
        self.PriorNet = PriorNet(
            in_channels=cfg.PriorNet_input_channel,
            out_channels=cfg.PriorNet_output_channel,
            feat_dim=cfg.PriorNet_feat_dim,
            hyperspectral_channels=cfg.PriorNet_SPM_dim,
            num_stages=cfg.PriorNet_num_stages,
        ).to(self.device)

        self.recon_loss = nn.SmoothL1Loss()
        self.opt_P, self.sch_P = init_optimizer(
            self.PriorNet.parameters(),
            lr=cfg.PriorNet_lr,
            optimizer_type=cfg.PriorNet_opt,
        )
        print("=" * 60)

        # --------------------------------------------------------------
        # Discriminator
        # --------------------------------------------------------------
        print("=" * 60)
        print("Initializing Discriminator")
        self.Discriminator = Discriminator(
            in_channels=cfg.Discriminator_in_channel,
            hidden_channels=cfg.Discriminator_hidden_channels,
        ).to(self.device)

        self.opt_D, self.sch_D = init_optimizer(
            self.Discriminator.parameters(),
            lr=cfg.Discriminator_lr,
            optimizer_type=cfg.Discriminator_opt,
        )
        print("=" * 60)

        # --------------------------------------------------------------
        # UALNet
        # --------------------------------------------------------------
        print("=" * 60)
        print("Initializing UALNet")
        self.UALNet = UALNet(
            lambda_1=cfg.UALNet_lambda_1,
            lambda_2=cfg.UALNet_lambda_2,
            mu=cfg.UALNet_mu,
            dim=cfg.UALNet_output_channel,
            s_channel=cfg.UALNet_input_channel,
            num_iter=cfg.UALNet_num_iter,
        ).to(self.device)

        self.opt_U, self.sch_U = init_optimizer(
            self.UALNet.parameters(),
            lr=cfg.UALNet_lr,
            optimizer_type=cfg.UALNet_opt,
        )
        print("=" * 60)

        # --------------------------------------------------------------
        # Best-checkpoint tracking for PriorNet
        # --------------------------------------------------------------
        self.best_priornet_epoch = -1
        self.best_priornet_psnr = -float("inf")
        self.best_priornet_sam = float("inf")
        

    def sam_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Spectral Angle Mapper (SAM) loss.

        Args:
            pred: Predicted tensor of shape (B, C, H, W).
            target: Ground-truth tensor of shape (B, C, H, W).

        Returns:
            Scalar SAM loss.
        """
        pred_norm = torch.norm(pred, p="fro", dim=1)
        target_norm = torch.norm(target, p="fro", dim=1)

        cosine = torch.sum(pred * target, dim=1) / (pred_norm * target_norm + 1e-12)
        cosine = torch.clamp(cosine, min=-1.0 + 1e-7, max=1.0 - 1e-7)

        loss = torch.arccos(cosine).mean()
        return loss
    
    def priornet_criterion(
        self,
        model_output: torch.Tensor,
        aviris: torch.Tensor,
        prior: torch.Tensor,
        sen_simu: torch.Tensor,
    ) -> torch.Tensor:
        """
        Composite loss for PriorNet training.

        Args:
            model_output:
                PriorNet reconstructed output,
                shape (B, 12, H, W).

            aviris:
                Real hyperspectral image used to construct the ground-truth
                spectral prior matrix,
                shape (B, 186, H, W).

            prior:
                Predicted spectral prior matrix from PriorNet,
                shape (B, 186, 186).

            sen_simu:
                Ground-truth spatial-resolution-unified Sentinel-2 image,
                shape (B, 12, H, W).

        Returns:
            Scalar loss tensor.
        """
        prior_gt = rearrange(aviris, "b c h w -> b c (h w)")
        prior_gt = prior_gt @ prior_gt.permute(0, 2, 1)

        fake_fft = torch.fft.fft2(model_output, dim=(-2, -1))
        real_fft = torch.fft.fft2(sen_simu, dim=(-2, -1))

        loss_recon = self.recon_loss(model_output, sen_simu)
        loss_sam = self.sam_loss(model_output, sen_simu)
        loss_fft = self.recon_loss(fake_fft.abs(), real_fft.abs())
        loss_prior = self.recon_loss(prior, prior_gt)

        total_loss = (
            loss_recon
            + 2.5e-3 * loss_sam
            + 2.5e-3 * loss_fft
            + 5e-4 * loss_prior
        )
        return total_loss

    def priornet_optimization(
        self,
        sentinel_2: torch.Tensor,
        aviris: torch.Tensor,
        sen_simu: torch.Tensor,
    ) -> torch.Tensor:
        """
        One optimization step for PriorNet.

        Args:
            sentinel_2:
                Multi-resolution Sentinel-2 input,
                shape (B, 12, H/2, W/2).

            aviris:
                Real hyperspectral image,
                shape (B, 186, H, W).

            sen_simu:
                Ground-truth spatial-resolution-unified Sentinel-2 image,
                shape (B, 12, H, W).

        Returns:
            Scalar PriorNet training loss.
        """
        self.PriorNet.train()
        self.opt_P.zero_grad()

        model_output, prior = self.PriorNet(sentinel_2)

        loss = self.priornet_criterion(
            model_output=model_output,
            aviris=aviris,
            prior=prior,
            sen_simu=sen_simu,
        )

        loss.backward()
        self.opt_P.step()

        return loss

    def train_priornet_one_epoch(self, epoch: int) -> float:

        train_loss_list = []

        for batch_data in self.train_loader:
            sentinel_2 = batch_data["Sentinel_2"].to(self.device)
            aviris = batch_data["AVIRIS"].to(self.device)
            sen_simu = batch_data["sen_simu"].to(self.device)

            loss = self.priornet_optimization(
                sentinel_2=sentinel_2,
                aviris=aviris,
                sen_simu=sen_simu,
            )
            train_loss_list.append(loss.item())

            msg = f"[Epoch {epoch}] PriorNet Loss: {loss.item():.6f}"
            #print(msg)

            with open(join(cfg.save_path, "msg_files_everyloss"), "a+") as f:
                f.write(msg + "\n")

        self.sch_P.step()

        train_loss = float(np.mean(train_loss_list))
        return train_loss


    def test_priornet(self, epoch: int):
        """
        Evaluate PriorNet on the test set.

        Returns:
            avg_psnr (float)
            avg_sam (float)
        """
        self.PriorNet.eval()

        psnr_list = []
        sam_list = []

        with torch.no_grad():
            for batch_data in self.test_loader:
                sentinel_2 = batch_data["Sentinel_2"].to(self.device)
                sen_simu = batch_data["sen_simu"].to(self.device)
                #file_name = batch_data["fname"][0]

                model_output, _ = self.PriorNet(sentinel_2)

                pred_np = model_output.squeeze(0).permute(1, 2, 0).cpu().numpy()
                gt_np = sen_simu.squeeze(0).permute(1, 2, 0).cpu().numpy()

                current_psnr = psnr(gt_np.transpose(2, 0, 1), pred_np.transpose(2, 0, 1))
                current_sam = sam(gt_np.transpose(2, 0, 1), pred_np.transpose(2, 0, 1))

                psnr_list.append(current_psnr)
                sam_list.append(current_sam)

        avg_psnr = float(np.mean(psnr_list))
        avg_sam = float(np.mean(sam_list))

        msg = f"[Epoch {epoch}] PriorNet Test | PSNR: {avg_psnr:.6f} | SAM: {avg_sam:.6f}"
        print(msg)

        with open(self.epoch_log_path, "a+") as f:
            f.write(msg + "\n")

        return avg_psnr, avg_sam

    def _update_best_priornet(self, epoch: int, avg_psnr: float, avg_sam: float) -> bool:
        """
        Update best PriorNet checkpoint.

        Selection rule:
            - Higher PSNR is better
            - If PSNR ties, lower SAM is better
        """
        is_better = False

        if avg_psnr > self.best_priornet_psnr:
            is_better = True
        elif np.isclose(avg_psnr, self.best_priornet_psnr) and avg_sam < self.best_priornet_sam:
            is_better = True

        if is_better:
            self.best_priornet_epoch = epoch
            self.best_priornet_psnr = avg_psnr
            self.best_priornet_sam = avg_sam

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": self.PriorNet.state_dict(),
                    "optimizer_state_dict": self.opt_P.state_dict(),
                    "scheduler_state_dict": self.sch_P.state_dict(),
                    "best_psnr": avg_psnr,
                    "best_sam": avg_sam,
                },
                self.best_priornet_path,
            )

            msg = (
                f"[Best PriorNet Updated] Epoch: {epoch} | "
                f"PSNR: {avg_psnr:.6f} | SAM: {avg_sam:.6f}"
            )
            print(msg)

            with open(self.epoch_log_path, "a+") as f:
                f.write(msg + "\n")

        return is_better

    def load_best_priornet(self) -> None:
        """
        Load the best PriorNet checkpoint after training.
        """
        if not os.path.isfile(self.best_priornet_path):
            raise FileNotFoundError(
                f"Best PriorNet checkpoint not found: {self.best_priornet_path}"
            )

        checkpoint = torch.load(self.best_priornet_path, map_location=self.device)
        self.PriorNet.load_state_dict(checkpoint["model_state_dict"])

        print("=" * 60)
        print(
            f"Loaded best PriorNet from epoch {checkpoint['epoch']} "
            f"(PSNR={checkpoint['best_psnr']:.6f}, SAM={checkpoint['best_sam']:.6f})"
        )
        print("=" * 60)

    def train_priornet(self) -> None:
        """
        Full PriorNet training pipeline.
        """
        print("=" * 60)
        print("PriorNet Training Begin")
        print("=" * 60)

        for epoch in tqdm(range(cfg.Prior_epoch_num)):
            train_loss = self.train_priornet_one_epoch(epoch)

            msg = f"[Epoch {epoch}] PriorNet Train Loss: {train_loss:.6f}"
            print(msg)

            with open(self.epoch_log_path, "a+") as f:
                f.write(msg + "\n")

            # Periodic evaluation
            if epoch % cfg.PriorNet_test_period == 0:
                print("\nPriorNet Testing")
                avg_psnr, avg_sam = self.test_priornet(epoch)

                self._update_best_priornet(
                    epoch=epoch,
                    avg_psnr=avg_psnr,
                    avg_sam=avg_sam,
                )

                # optional regular checkpoint
                torch.save(
                    {"model_state_dict": self.PriorNet.state_dict(), "epoch": epoch},
                    join(cfg.save_path, f"PriorNet_epoch_{epoch}.pth"),
                )
                print("-" * 50)

        print("=" * 60)
        print(
            f"PriorNet training finished. Best epoch: {self.best_priornet_epoch} | "
            f"Best PSNR: {self.best_priornet_psnr:.6f} | "
            f"Best SAM: {self.best_priornet_sam:.6f}"
        )
        print(f"Best checkpoint saved to: {self.best_priornet_path}")
        print("=" * 60)

        # Automatically load best PriorNet for subsequent training
        self.load_best_priornet()
    
      
    def ualnet_criterion(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Weighted SAM-like reconstruction loss for UALNet output.

        Args:
            pred: Predicted hyperspectral image, shape (B, C, H, W).
            target: Ground-truth hyperspectral image, shape (B, C, H, W).

        Returns:
            Scalar loss tensor.
        """
        norm_pred = torch.norm(pred, p="fro", dim=1, keepdim=True)
        norm_target = torch.norm(target, p="fro", dim=1, keepdim=True)

        cosine = torch.sum(pred * target, dim=1, keepdim=True) / (norm_pred * norm_target + 1e-12)
        #cosine = torch.clamp(cosine, min=-1.0 + 1e-7, max=1.0 - 1e-7)

        spectral_angle = torch.arccos(cosine)
        loss = spectral_angle * (pred - target).abs()

        return loss.mean()

    # --------------------------------------------------------------
    # One step of alternating training
    # --------------------------------------------------------------
    def train_ualnet_step(
        self,
        s_u: torch.Tensor,
        real_hsi: torch.Tensor,
        p: torch.Tensor,
        train_generator: bool,
    ):
        """
        One optimization step for either UALNet or Discriminator.

        Args:
            s_u: PriorNet output, shape (B, 12, H, W).
            real_hsi: Ground-truth AVIRIS hyperspectral image, shape (B, 186, H, W).
            p: Spectral prior matrix, shape (B, 186, 186).
            train_generator: If True, optimize UALNet. Otherwise optimize Discriminator.

        Returns:
            err_g, err_d, d_real_mean, d_fake_mean
        """
        if train_generator:
            self.UALNet.train()
            self.Discriminator.eval()
        else:
            self.UALNet.eval()
            self.Discriminator.train()

        self.opt_U.zero_grad()
        self.opt_D.zero_grad()

        discriminator_for_unfolding = copy.deepcopy(self.Discriminator)
        discriminator_for_unfolding.eval()

        fake_hsi = self.UALNet(s_u, discriminator_for_unfolding, p)

        out_d_real = self.Discriminator(real_hsi)
        out_d_fake = self.Discriminator(fake_hsi.detach())

        err_d = (
            -torch.log(out_d_real.mean() + 1e-12)
            - torch.log(1.0 - out_d_fake.mean() + 1e-12)
        )

        err_g = self.ualnet_criterion(fake_hsi, real_hsi)

        if train_generator:
            err_g.backward()
            self.opt_U.step()
        else:
            err_d.backward()
            self.opt_D.step()

        return err_g, err_d, out_d_real.mean(), out_d_fake.mean()

    def crop(self, data, data2, crop_size=(64, 64), enable_crop=True):
   
        if not enable_crop:
            return data, data2

        if data.ndim != 4 or data2.ndim != 4:
            raise ValueError(
                f"Input tensors must be 4D (B,C,H,W). "
                f"Got {data.shape} and {data2.shape}"
            )

        if data.shape[0] != data2.shape[0]:
            raise ValueError("Batch size mismatch between data and data2")

        if data.shape[-2:] != data2.shape[-2:]:
            raise ValueError(
                f"Spatial size mismatch: {data.shape[-2:]} vs {data2.shape[-2:]}"
            )

        B, C, H, W = data.shape
        crop_h, crop_w = crop_size

        if crop_h > H or crop_w > W:
            print(
                f"[Warning] Crop size {crop_size} larger than image size {(H,W)}. "
                "Return original tensors."
            )
            return data, data2

        if crop_h == H and crop_w == W:
            return data, data2

        h_idx = np.random.randint(0, H - crop_h + 1)
        w_idx = np.random.randint(0, W - crop_w + 1)

        cropped_data = data[:, :, h_idx:h_idx + crop_h, w_idx:w_idx + crop_w]
        cropped_data2 = data2[:, :, h_idx:h_idx + crop_h, w_idx:w_idx + crop_w]

        return cropped_data, cropped_data2
    
    def train_ualnet_one_epoch(self, epoch: int, train_generator: bool):
        train_loss_g_list = []
        train_loss_d_list = []
        train_real_prob_list = []
        train_fake_prob_list = []

        for batch_data in self.train_loader:
            sentinel_2 = batch_data["Sentinel_2"].to(self.device)
            aviris = batch_data["AVIRIS"].to(self.device)

            with torch.no_grad():
                s_u, p = self.PriorNet(sentinel_2)

            if epoch<200:
                enable_crop = True
            else:
                enable_crop = False

            s_u,aviris = self.crop(s_u, aviris, crop_size=(64, 64), enable_crop=enable_crop)

            err_g, err_d, d_real, d_fake = self.train_ualnet_step(
                s_u=s_u,
                real_hsi=aviris,
                p=p,
                train_generator=train_generator,
            )

            train_loss_g_list.append(err_g.item())
            train_loss_d_list.append(err_d.item())
            train_real_prob_list.append(d_real.item())
            train_fake_prob_list.append(d_fake.item())

        self.sch_U.step()
        self.sch_D.step()

        train_loss_g = float(np.mean(train_loss_g_list))
        train_loss_d = float(np.mean(train_loss_d_list))
        train_real_prob = float(np.mean(train_real_prob_list))
        train_fake_prob = float(np.mean(train_fake_prob_list))

        return train_loss_g, train_loss_d, train_real_prob, train_fake_prob

    # --------------------------------------------------------------
    # Testing
    # --------------------------------------------------------------
    def test_ualnet(self, epoch: int):
        """
        Evaluate UALNet on the test set.

        Returns:
            avg_psnr, avg_sam
        """
        self.PriorNet.eval()
        self.UALNet.eval()
        self.Discriminator.eval()

        psnr_list = []
        sam_list = []

        with torch.enable_grad():
            for batch_data in self.test_loader:
                sentinel_2 = batch_data["Sentinel_2"].to(self.device)
                aviris = batch_data["AVIRIS"].to(self.device)
                with torch.no_grad():
                    s_u, p = self.PriorNet(sentinel_2)
                pred_hsi = self.UALNet(s_u, self.Discriminator, p)

                pred_np = pred_hsi.squeeze(0).detach().permute(1, 2, 0).cpu().numpy()
                gt_np = aviris.squeeze(0).permute(1, 2, 0).cpu().numpy()

                current_psnr = psnr(gt_np.transpose(2, 0, 1), pred_np.transpose(2, 0, 1))
                current_sam = sam(gt_np.transpose(2, 0, 1), pred_np.transpose(2, 0, 1))

                psnr_list.append(current_psnr)
                sam_list.append(current_sam)

        avg_psnr = float(np.mean(psnr_list))
        avg_sam = float(np.mean(sam_list))

        msg = f"[Epoch {epoch}] UALNet Test | PSNR: {avg_psnr:.6f} | SAM: {avg_sam:.6f}"
        print(msg)

        with open(self.epoch_log_path, "a+") as f:
            f.write(msg + "\n")

        return avg_psnr, avg_sam

    # --------------------------------------------------------------
    # Validation
    # --------------------------------------------------------------
    def validate_final_ualnet(self):
        """
        Final validation after all best checkpoints are loaded.

        Returns:
            avg_psnr, avg_sam
        """
        self.PriorNet.eval()
        self.UALNet.eval()
        self.Discriminator.eval()

        psnr_list = []
        sam_list = []

        with torch.enable_grad():
            for batch_data in self.val_loader:
                sentinel_2 = batch_data["Sentinel_2"].to(self.device)
                aviris = batch_data["AVIRIS"].to(self.device)
                with torch.no_grad():
                    s_u, p = self.PriorNet(sentinel_2)
                pred_hsi = self.UALNet(s_u, self.Discriminator, p)

                pred_np = pred_hsi.squeeze(0).detach().permute(1, 2, 0).cpu().numpy()
                gt_np = aviris.squeeze(0).permute(1, 2, 0).cpu().numpy()

                current_psnr = psnr(gt_np.transpose(2, 0, 1), pred_np.transpose(2, 0, 1))
                current_sam = sam(gt_np.transpose(2, 0, 1), pred_np.transpose(2, 0, 1))

                psnr_list.append(current_psnr)
                sam_list.append(current_sam)

        avg_psnr = float(np.mean(psnr_list))
        avg_sam = float(np.mean(sam_list))

        print("=" * 60)
        print(f"Final Validation | PSNR: {avg_psnr:.6f} | SAM: {avg_sam:.6f}")
        print("=" * 60)

        with open(self.epoch_log_path, "a+") as f:
            f.write(f"Final Validation | PSNR: {avg_psnr:.6f} | SAM: {avg_sam:.6f}\n")

        return avg_psnr, avg_sam

    # --------------------------------------------------------------
    # Best checkpoint manager
    # --------------------------------------------------------------
    def _update_best_ualnet(self, epoch: int, avg_psnr: float, avg_sam: float) -> bool:
        """
        Update best UALNet / Discriminator checkpoint.
        """
        is_better = False

        if avg_psnr > self.best_ualnet_psnr:
            is_better = True
        elif np.isclose(avg_psnr, self.best_ualnet_psnr) and avg_sam < self.best_ualnet_sam:
            is_better = True

        if is_better:
            self.best_ualnet_epoch = epoch
            self.best_ualnet_psnr = avg_psnr
            self.best_ualnet_sam = avg_sam

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": self.UALNet.state_dict(),
                    "optimizer_state_dict": self.opt_U.state_dict(),
                    "scheduler_state_dict": self.sch_U.state_dict(),
                    "best_psnr": avg_psnr,
                    "best_sam": avg_sam,
                },
                self.best_ualnet_path,
            )

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": self.Discriminator.state_dict(),
                    "optimizer_state_dict": self.opt_D.state_dict(),
                    "scheduler_state_dict": self.sch_D.state_dict(),
                },
                self.best_discriminator_path,
            )

            msg = (
                f"[Best UALNet Updated] Epoch: {epoch} | "
                f"PSNR: {avg_psnr:.6f} | SAM: {avg_sam:.6f}"
            )
            print(msg)

            with open(self.epoch_log_path, "a+") as f:
                f.write(msg + "\n")

        return is_better

    def load_best_ualnet_and_discriminator(self) -> None:
        if not os.path.isfile(self.best_ualnet_path):
            raise FileNotFoundError(f"Best UALNet checkpoint not found: {self.best_ualnet_path}")
        if not os.path.isfile(self.best_discriminator_path):
            raise FileNotFoundError(
                f"Best Discriminator checkpoint not found: {self.best_discriminator_path}"
            )

        ual_ckpt = torch.load(self.best_ualnet_path, map_location=self.device)
        d_ckpt = torch.load(self.best_discriminator_path, map_location=self.device)

        self.UALNet.load_state_dict(ual_ckpt["model_state_dict"])
        self.Discriminator.load_state_dict(d_ckpt["model_state_dict"])

        print("=" * 60)
        print(
            f"Loaded best UALNet/Discriminator from epoch {ual_ckpt['epoch']} "
            f"(PSNR={ual_ckpt['best_psnr']:.6f}, SAM={ual_ckpt['best_sam']:.6f})"
        )
        print("=" * 60)

    # --------------------------------------------------------------
    # Full UALNet training
    # --------------------------------------------------------------
    def train_ualnet(self) -> None:
        """
        Full training pipeline for UALNet + Discriminator.
        """
        
        self.PriorNet.eval()
        for p in self.PriorNet.parameters():
            p.requires_grad = False

        self.best_ualnet_epoch = -1
        self.best_ualnet_psnr = -float("inf")
        self.best_ualnet_sam = float("inf")

        self.best_ualnet_path = join(cfg.save_path, "UALNet_the_best.pth")
        self.best_discriminator_path = join(cfg.save_path, "Discriminator_the_best.pth")

        print("=" * 60)
        print("UALNet Training Begin")
        print("=" * 60)

        train_generator = True
        switch_flag = 0

        for epoch in tqdm(range(cfg.UALNet_epoch_num)):
            if epoch <= 250:
                if epoch % cfg.UALNet_period == 0 and epoch != 0:
                    switch_flag += cfg.UALNet_period
                    if (switch_flag // cfg.UALNet_period) % 2 == 0:
                        train_generator = True
                    else:
                        train_generator = False
            else:
                train_generator = True

            model_flag = 1 if train_generator else 0

            if epoch == 250:
                for param_group in self.opt_U.param_groups:
                    param_group["lr"] = 8e-5

            train_loss_g, train_loss_d, d_real, d_fake = self.train_ualnet_one_epoch(epoch,train_generator)

            msg = (
                f"[Epoch {epoch}] "
                f"TrainUALNet: {model_flag} | "
                f"Train_UALNet_Loss: {train_loss_g:.6f} | "
                f"Train_D_Loss: {train_loss_d:.6f} | "
                f"Real_Prob: {d_real:.6f} | "
                f"Fake_Prob: {d_fake:.6f}"
            )
            print(msg)

            with open(self.epoch_log_path, "a+") as f:
                f.write(msg + "\n")

            if epoch % cfg.UALNet_test_period == 0:
                torch.save(
                    {"model_state_dict": self.UALNet.state_dict(), "epoch": epoch},
                    join(cfg.save_path, f"UALNet_epoch_{epoch}.pth"),
                )
                torch.save(
                    {"model_state_dict": self.Discriminator.state_dict(), "epoch": epoch},
                    join(cfg.save_path, f"Discriminator_epoch_{epoch}.pth"),
                )

                print("\nUALNet Testing")
                avg_psnr, avg_sam = self.test_ualnet(epoch)

                self._update_best_ualnet(
                    epoch=epoch,
                    avg_psnr=avg_psnr,
                    avg_sam=avg_sam,
                )
                print("-" * 50)

        print("=" * 60)
        print(
            f"UALNet training finished. Best epoch: {self.best_ualnet_epoch} | "
            f"Best PSNR: {self.best_ualnet_psnr:.6f} | "
            f"Best SAM: {self.best_ualnet_sam:.6f}"
        )
        print(f"Best UALNet checkpoint saved to: {self.best_ualnet_path}")
        print(f"Best Discriminator checkpoint saved to: {self.best_discriminator_path}")
        print("=" * 60)

        # Load best checkpoints automatically
        self.load_best_ualnet_and_discriminator()

       
    
    def train_overall(self) ->None:
        self.best_priornet_path = join(cfg.save_path, "PriorNet_the_best.pth")
        self.best_ualnet_path = join(cfg.save_path, "UALNet_the_best.pth")
        self.best_discriminator_path = join(cfg.save_path, "Discriminator_the_best.pth")

        if os.path.isfile(self.best_priornet_path):

            print("=" * 60)
            print("Best PriorNet checkpoints found.")
            print("Skipping PriorNet training.")
            print("=" * 60)

            self.load_best_priornet()
        else:
            self.train_priornet()

        if os.path.isfile(self.best_ualnet_path) and os.path.isfile(self.best_discriminator_path):

            print("=" * 60)
            print("Best UALNet and Discriminator checkpoints found.")
            print("Skipping training and running final validation.")
            print("=" * 60)

            self.load_best_ualnet_and_discriminator()
        else:
            self.train_priornet()
        

        # Final validation with best PriorNet + best UALNet + best Discriminator
        self.validate_final_ualnet()

            
if __name__ == "__main__" :
    model = Trainer()
    model.train_overall()
