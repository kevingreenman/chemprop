import torch
from torch import Tensor
from torch.nn import functional as F

from chemprop.v2.models.models.base import MPNN
from chemprop.v2.models.models.multifidelity import MultifidelityMPNN


class RegressionMPNN(MPNN):
    _DATASET_TYPE = "regression"
    _DEFAULT_CRITERION = "mse"
    _DEFAULT_METRIC = "rmse"


class MultifidelityRegressionMPNN(MultifidelityMPNN):
    _DATASET_TYPE = "regression"
    _DEFAULT_CRITERION = "mse"
    _DEFAULT_METRIC = "rmse"


class MveRegressionMPNN(RegressionMPNN):
    _DEFAULT_CRITERION = "mve"

    @property
    def n_targets(self) -> int:
        return 2

    def forward(self, inputs, X_f) -> Tensor:
        Y = super().forward(inputs, X_f=X_f)

        Y_mean, Y_var = Y.split(Y.shape[1] // 2, 1)
        Y_var = F.softplus(Y_var)

        return torch.cat((Y_mean, Y_var), 1)

    def predict_step(self, *args, **kwargs) -> tuple[Tensor, ...]:
        Y = super().predict_step(*args, **kwargs)[0]
        Y_mean, Y_var = Y.split(Y.shape[1] // 2, dim=1)

        return Y_mean, Y_var


class EvidentialMPNN(RegressionMPNN):
    _DEFAULT_CRITERION = "evidential"

    @property
    def n_targets(self) -> int:
        return 4

    def forward(self, inputs, X_f) -> Tensor:
        Y = super().forward(inputs, X_f)

        means, lambdas, alphas, betas = Y.split(Y.shape[1] // 4, dim=1)
        lambdas = F.softplus(lambdas)
        alphas = F.softplus(alphas) + 1
        betas = F.softplus(betas)

        return torch.cat((means, lambdas, alphas, betas), 1)

    def predict_step(self, *args, **kwargs) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        Y = super().predict_step(*args, **kwargs)[0]
        means, lambdas, alphas, betas = Y.split(Y.shape[1] // 4, 1)

        return means, lambdas, alphas, betas

class EvidentialMultifidelityMPNN(MultifidelityRegressionMPNN):

    _DEFAULT_CRITERION = "evidential"

    def __init__(self, mpn_block, n_tasks=1, mpn_block_low_fidelity=None, init_lr=0.1, max_lr=0.5, final_lr=0.001, loss_mod=1):
        super().__init__(mpn_block=mpn_block, 
                         mpn_block_low_fidelity=mpn_block_low_fidelity, 
                         n_tasks=n_tasks, 
                         init_lr=init_lr, 
                         max_lr=max_lr, 
                         final_lr=final_lr,
                         loss_mod=loss_mod)

        self.ffn_high_fidelity = self.build_ffn(
            mpn_block.output_dim + 1,  # +1 for LF preds, +3 for Uncertainty Features
            1,
        )
        self.ffn_low_fidelity = self.build_ffn(
            mpn_block.output_dim,
            4,
        )

        self.mpn_block = mpn_block
        self.mpn_block_low_fidelity = mpn_block_low_fidelity
        self.loss_mod = loss_mod

    def forward(self, inputs, X_f = None):
        if self.mpn_block_low_fidelity is not None:
            lf_output = self.ffn_low_fidelity(self.mpn_block_low_fidelity(*inputs))
        else:
            lf_output = self.ffn_low_fidelity(self.mpn_block(*inputs))

        means, lambdas, alphas, betas = lf_output.split(lf_output.shape[1] // 4, dim=1)
        lambdas = F.softplus(lambdas)
        alphas = F.softplus(alphas) + 1
        betas = F.softplus(betas)

        alphas = torch.clamp(alphas, min=1.0001)
        betas = torch.clamp(betas, min=0.0001)
        lambdas = torch.clamp(lambdas, min=0.0001)

        al_u = betas/(alphas-1)
        ep_u = al_u/lambdas

        extra_features = torch.cat((means, al_u, ep_u), 1)
        hf_output = self.ffn_high_fidelity(self.fingerprint(inputs, X_f=means))

        return (hf_output, lf_output)


    def training_step(self, batch, batch_idx):
        bmg, X_vd, features, targets, weights, lt_targets, gt_targets = batch

        mask_hf = targets[:, 1].isfinite()
        mask_lf = targets[:, 0].isfinite()

        hf_output_i, lf_output_i = self.forward((bmg, X_vd))

        lf_loss = self.criterion.calc(lf_output_i[mask_lf], targets[:, 0][mask_lf].reshape(-1, 1)).sum()
        hf_loss = F.mse_loss(hf_output_i[mask_hf], targets[:, 1][mask_hf].reshape(-1, 1)).sum()

        total_loss = self.loss_mod*lf_loss + hf_loss

        #self.log("LF train/loss", lf_loss, prog_bar=True)
        #51self.log("HF train/loss", hf_loss, prog_bar=True)

        return total_loss
    
    
class MveMultifidelity(MultifidelityRegressionMPNN):

    _DEFAULT_CRITERION = "mve"

    def __init__(self, mpn_block, n_tasks=1, mpn_block_low_fidelity=None, loss_mod=1):
        super().__init__(mpn_block=mpn_block, 
                         mpn_block_low_fidelity=mpn_block_low_fidelity, 
                         n_tasks=n_tasks, 
                         init_lr=0.01, 
                         max_lr=0.05, 
                         final_lr=0.01,
                         loss_mod=loss_mod)

        self.ffn_high_fidelity = self.build_ffn(
            mpn_block.output_dim + 2,  # +1 for LF preds, +2 for Uncertainty Features
            1,
        )
        self.ffn_low_fidelity = self.build_ffn(
            mpn_block.output_dim,
            2,
        )

        self.mpn_block = mpn_block
        self.mpn_block_low_fidelity = mpn_block_low_fidelity
        self.loss_mod = loss_mod

    def forward(self, inputs, X_f = None):
        if self.mpn_block_low_fidelity is not None:
            lf_output = self.ffn_low_fidelity(self.mpn_block_low_fidelity(*inputs))
        else:
            lf_output = self.ffn_low_fidelity(self.mpn_block(*inputs))

        means, var = lf_output.split(lf_output.shape[1] // 2, dim=1)

        extra_features = torch.cat((means, var), 1)
        hf_output = self.ffn_high_fidelity(self.fingerprint(inputs, X_f=extra_features))

        return (hf_output, lf_output)


    def training_step(self, batch, batch_idx):
        bmg, X_vd, features, targets, weights, lt_targets, gt_targets = batch

        mask_hf = targets[:, 1].isfinite()
        mask_lf = targets[:, 0].isfinite()

        hf_output_i, lf_output_i = self.forward((bmg, X_vd))
        #print(hf_output_i, lf_output_i)

        lf_loss = self.criterion.calc(lf_output_i[mask_lf], targets[:, 0][mask_lf].reshape(-1, 1)).sum()
        hf_loss = F.mse_loss(hf_output_i[mask_hf], targets[:, 1][mask_hf].reshape(-1, 1)).sum()
        #print(f'HF: {hf_loss}, LF: {lf_loss}')

        total_loss = self.loss_mod*lf_loss + hf_loss
        #print(lf_loss, hf_loss)
        
        #self.log("LF train/loss", lf_loss, prog_bar=True)
        #self.log("HF train/loss", hf_loss, prog_bar=True)

        return total_loss


class EvidentialDualMF(EvidentialMultifidelityMPNN):

    _DEFAULT_CRITERION = "evidential"

    @property
    def n_targets(self) -> int:
        return 4
    
    def __init__(self, mpn_block, n_tasks=4, mpn_block_low_fidelity=None):
        super().__init__(mpn_block=mpn_block, 
                         mpn_block_low_fidelity=mpn_block_low_fidelity, 
                         n_tasks=n_tasks,
                         init_lr=0.01, 
                         max_lr=0.05, 
                         final_lr=0.001)

        self.ffn_high_fidelity = self.build_ffn(
            mpn_block.output_dim + 3,  # +1 for LF preds, +2 for Uncertainty Features
            4,
        )
        self.ffn_low_fidelity = self.build_ffn(
            mpn_block.output_dim,
            4,
        )

        self.mpn_block = mpn_block
        self.mpn_block_low_fidelity = mpn_block_low_fidelity


    def training_step(self, batch, batch_idx):
        bmg, X_vd, features, targets, weights, lt_targets, gt_targets = batch

        mask_hf = targets[:, 1].isfinite()
        mask_lf = targets[:, 0].isfinite()

        hf_output_i, lf_output_i = self.forward((bmg, X_vd))

        lf_loss = self.criterion.calc(lf_output_i[mask_lf], targets[:, 0][mask_lf].reshape(-1, 1)).sum()
        hf_loss = self.criterion.calc(hf_output_i[mask_hf], targets[:, 1][mask_hf].reshape(-1, 1)).sum()

        l_mod = 1
        total_loss = l_mod*lf_loss + hf_loss
        #print(lf_loss, hf_loss)
        #mve lf
        
        self.log("LF train/loss", lf_loss, prog_bar=True)
        self.log("HF train/loss", hf_loss, prog_bar=True)

        return total_loss
    
    #FIX
    def predict_step(self, *args, **kwargs) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        Y = super().forward(*args, **kwargs)[0]
        means, lambdas, alphas, betas = Y.split(Y.shape[1] // 4, 1)

        return ((means, lambdas, alphas, betas),)
