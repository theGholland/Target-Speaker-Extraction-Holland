import torch


def get_device() -> torch.device:
    """Select appropriate computation device and announce it."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA device: {torch.cuda.get_device_name(device)}")
        return device
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device")
        return device
    raise RuntimeError("No CUDA or MPS device available")
