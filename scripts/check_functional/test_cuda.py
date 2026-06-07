import torch


DEVICE = "cuda"


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    x = torch.arange(6, dtype=torch.float32, device=DEVICE).reshape(2, 3)
    w = torch.ones((3, 1), dtype=torch.float32, device=DEVICE)
    y = x @ w

    print(f"device={x.device}")
    print(f"cuda_device_name={torch.cuda.get_device_name(x.device)}")
    print(f"x={x.cpu().numpy().tolist()}")
    print(f"y={y.cpu().numpy().reshape(-1).tolist()}")


if __name__ == "__main__":
    main()
