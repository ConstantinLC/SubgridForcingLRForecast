import torch
from torch import nn
import pytorch_lightning as pl
import torch.optim as optim
from parametrization.subgrid_parametrization import SubgridParametrization
from forecast.unet import UNet2d, UNet2d_hr_encoder
from torchmetrics.regression import PearsonCorrCoef

class ForecastModel(pl.LightningModule):
    """
    A PyTorch Lightning model for forecasting, incorporating options for high-resolution encoding
    and subgrid parametrization.
    """

    def __init__(
        self,
        n_input_scalar_components: int,
        n_output_scalar_components: int,
        learning_rate: float = 1e-3,
        add_highres_encoding: bool = False,
        add_parametrization: bool = False,
        autoregressive: bool = False,
        noise_level: float = 0,
    ) -> None:
        super().__init__()
        self.n_input_scalar_components = n_input_scalar_components
        self.n_output_scalar_components = n_output_scalar_components
        self.add_highres_encoding = add_highres_encoding
        self.add_parametrization = add_parametrization
        self.learning_rate = learning_rate
        self.autoregressive = autoregressive
        self.noise_level = noise_level

        # Initialize pre-forecast encoding model based on configuration
        if self.add_highres_encoding:
            # Encoder for high-resolution input
            self.pre_forecast_encoding_model = UNet2d_hr_encoder(in_channels=2, out_channels=2, init_features=32)
        elif self.add_parametrization:
            # Subgrid parametrization model
            self.pre_forecast_encoding_model = SubgridParametrization(
                n_input_scalar_components=self.n_input_scalar_components,
                n_output_scalar_components=2
            )
        else:
            # Default UNet for low-resolution input if no specific encoding is added
            self.pre_forecast_encoding_model = UNet2d(in_channels=2, out_channels=2)

        # Main UNet for forecasting
        self.unet = UNet2d(in_channels=self.n_input_scalar_components, out_channels=2)

    def normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Normalizes the input tensor across specified dimensions."""
        return (tensor - torch.mean(tensor, axis=(0, 2, 3), keepdims=True)) / \
               (torch.std(tensor, axis=(0, 2, 3), keepdims=True) + 1e-8) # Added epsilon for numerical stability

    def forward(self, x: torch.Tensor, highres_x: torch.Tensor = None, noise: bool = False) -> torch.Tensor:
        """
        Forward pass of the model.
        Args:
            x: Low-resolution input tensor.
            highres_x: High-resolution input tensor, used if add_highres_encoding is True.
            noise: Whether to add Laplace noise to the pre-forecast encoding.
        Returns:
            The predicted output tensor.
        """
        if self.add_highres_encoding:
            pre_forecast_encoding = self.pre_forecast_encoding_model(highres_x)
        elif self.add_parametrization:
            pre_forecast_encoding = self.pre_forecast_encoding_model(x)
        else:
            pre_forecast_encoding = self.pre_forecast_encoding_model(x)

        if noise and self.noise_level != 0:
            distribution = torch.distributions.laplace.Laplace(loc=0, scale=self.noise_level)
            pre_forecast_encoding = pre_forecast_encoding + distribution.sample(
                sample_shape=pre_forecast_encoding.shape).to(x.device)

        # Currently, pre_forecast_encoding is computed but not used in the final `self.unet` call.
        # If it's meant to be concatenated with `x`, the line below should be used instead:
        # preds = self.unet(torch.cat((x, pre_forecast_encoding), dim=1))
        preds = self.unet(x) # Using only 'x' as input to the main UNet

        return preds

    def _step(self, batch: tuple, is_training: bool) -> torch.Tensor:
        """
        Helper function for training and validation steps.
        Args:
            batch: The current batch of data.
            is_training: Boolean indicating if it's a training step (influences noise addition).
        Returns:
            The calculated loss for the step.
        """
        if not self.autoregressive:
            input_lowres_q, target_lowres_q, _, input_highres_q, _ = batch

            input_lowres_q = self.normalize(input_lowres_q)
            target_lowres_q = self.normalize(target_lowres_q)
            input_highres_q = self.normalize(input_highres_q)

            preds = self.forward(input_lowres_q, highres_x=input_highres_q, noise=is_training)
            loss = nn.MSELoss()(preds, target_lowres_q)
        else:
            lowres_q, _, highres_q = batch

            preds = self.normalize(lowres_q[:, 0])
            loss = None
            for i in range(lowres_q.shape[1] - 1):
                if self.add_highres_encoding:
                    highres_q_i = self.normalize(highres_q[:, i])
                    preds = self.forward(preds, highres_x=highres_q_i)
                else:
                    preds = self.forward(preds)

                target = self.normalize(lowres_q[:, i + 1])
                current_loss = nn.MSELoss()(preds, target)
                if loss is None:
                    loss = current_loss
                else:
                    loss += current_loss
        return loss, preds, target_lowres_q if not self.autoregressive else target

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Performs a single training step."""
        loss, _, _ = self._step(batch, is_training=True)
        self.log('train/MSE', loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        """Performs a single validation step."""
        loss, preds, target = self._step(batch, is_training=False)
        
        # Log Pearson Correlation Coefficient only if not in autoregressive mode or if `target` is available
        # The original code calculates PearsonCorrCoef even in autoregressive mode on `target` which is only the last step's target
        # For a more meaningful metric in autoregressive mode, consider accumulating predictions and targets over the sequence.
        if not self.autoregressive:
            self.log('val/pearson_cor', PearsonCorrCoef(num_outputs=preds.shape[0]).to(preds.device)(
                torch.flatten(preds, start_dim=1).T, torch.flatten(target, start_dim=1).T).mean(), 
                on_step=False, on_epoch=True, prog_bar=False, sync_dist=True
            )
        self.log('val/MSE', loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        return loss

    def configure_optimizers(self):
        """Configures the optimizer for the model."""
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)
        return {'optimizer': optimizer}