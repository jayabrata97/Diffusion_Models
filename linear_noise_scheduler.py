import torch as T


class LinearNoiseScheduler:
    """
    Class for linear noise scheduler
    """

    def __init__(self, num_timesteps, beta_start, beta_end) -> None:
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.betas = T.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alpha_cum_prod = T.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_cum_prod = T.sqrt(self.alpha_cum_prod)
        self.sqrt_one_minus_alpha_cum_prod = T.sqrt(1 - self.alpha_cum_prod)

    def add_noise(self, original, noise, t) -> T.tensor:
        """Performs noise adding to the images
        Args:
            originial: Image on which noise is to be applied
            noise: Random noise tensor
            t: timestep of the forward process of shape -> (B,)
        Returns:
            noise added image
        """
        batch_size = original.shape[0]
        sqrt_alpha_cum_prod = self.sqrt_alpha_cum_prod.to(original.device)[t].reshape(
            batch_size
        )
        sqrt_one_minus_alpha_cum_prod = self.sqrt_one_minus_alpha_cum_prod.to(
            original.device
        )[t].reshape(batch_size)

        for _ in range(len(original.shape[0]) - 1):
            sqrt_alpha_cum_prod = sqrt_alpha_cum_prod.unsqueeze(-1)
        for _ in range(len(original.shape[0]) - 1):
            sqrt_one_minus_alpha_cum_prod = sqrt_one_minus_alpha_cum_prod.unsqueeze(-1)

        return (
            sqrt_alpha_cum_prod.to(original.device) * original
            + sqrt_one_minus_alpha_cum_prod.to(original.device) * noise
        )

    def sample_prev_timestep(self, x_t, noise_pred, t):
        """Use the noise prediction by model to get x_(t-1) using x_t and the noise predicted
        Args:
            x_t: Sample at current timestep
            noise_pred: Model noise prediction
            t: Current timestep
        Returns:
            Sampled image
        """
        x_0 = (
            x_t - (self.sqrt_one_minus_alpha_cum_prod.to(x_t.device)[t] * noise_pred)
        ) / T.sqrt(self.alpha_cum_prod.to(x_t.device)[t])
        x_0 = T.clamp(x_0, -1.0, 1.0)

        mean = x_t - ((self.betas.to(x_t.device)[t]) * noise_pred) / (
            self.sqrt_one_minus_alpha_cum_prod.to(x_t.device)[t]
        )
        mean = mean / T.sqrt(self.alphas.to(x_t.device)[t])

        if t == 0:
            return mean, x_0
        else:
            variance = (1 - self.alpha_cum_prod.to(x_t.device)[t - 1]) / (
                1.0 - self.alpha_cum_prod.to(x_t.device)[t]
            )
            variance = variance * self.betas.to(x_t.device)[t]
            sigma = variance**0.5
            z = T.randn(x_t.shape).to(x_t.device)

        return mean + sigma * z, x_0
