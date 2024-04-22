import torch
import torchvision
import argparse
import yaml
import os
from torchvision.utils import make_grid
from tqdm import tqdm
from model.UNet import UNet
from scheduler.linear_noise_scheduler import LinearNoiseScheduler

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sample(model, scheduler, train_config, model_config, diffusion_config):
    x_t = torch.randn(
        (
            train_config["num_samples"],
            model_config["pic_channels"],
            model_config["pic_size"],
            model_config["pic_size"],
        )
    ).to(device)

    for i in tqdm(reversed(range(diffusion_config["num_timesteps"]))):
        noise_pred = model(x_t, torch.as_tensor(i).unsqueeze(0).to(device))
        x_t, x_0_pred = scheduler.sample_prev_timestep(
            x_t, noise_pred, torch.as_tensor(i).to(device)
        )
        ims = torch.clamp(x_t, -1.0, 1.0).detach().cpu()
        ims = (ims + 1) / 2
        grid = make_grid(ims, nrow=train_config["num_grid_rows"])
        img = torchvision.transforms.ToPILImage()(grid)
        if not os.path.exists(os.path.join(train_config["task_name"], "samples")):
            os.mkdir(os.path.join(train_config["task_name"], "samples"))
        img.save(
            os.path.join(train_config["task_name"], "samples", "x0_{}.png".format(i))
        )
        img.close()


def infer(args):
    with open(args.config_path, "r") as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
        print(config)

    diffusion_config = config["diffudion_params"]
    model_config = config["model_params"]
    train_config = config["train_params"]

    model = UNet(model_config).to(device)
    model.load_state_dict(
        torch.load(
            os.path.join(train_config["task_name"], train_config["ckpt_name"]),
            map_location=device,
        )
    )
    model.eval()
    scheduler = LinearNoiseScheduler(
        num_timesteps=diffusion_config["num_timesteps"],
        beta_start=diffusion_config["beta_start"],
        beta_end=diffusion_config["beta_end"],
    )
    with torch.no_grad():
        sample(model, scheduler, train_config, model_config, diffusion_config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments for DDPM image generation")
    parser.add_argument(
        "--config", dest="config_path", default="config/default.yaml", type=str
    )
    args = parser.parse_args()
    infer(args)
