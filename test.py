'''
export CUDA_VISIBLE_DEVICES=1

'''

import time
import torch


DURATION_SECONDS = 30
MATRIX_SIZE = 4096


def main() -> None:
    if not torch.cuda.is_available():
        print("CUDA is not available.")
        return

    device = torch.device("cuda")
    device_name = torch.cuda.get_device_name(device)
    print(f"Running GPU load test on: {device_name}")
    print(f"Duration: {DURATION_SECONDS} seconds")

    torch.manual_seed(0)
    a = torch.randn(MATRIX_SIZE, MATRIX_SIZE, device=device)
    b = torch.randn(MATRIX_SIZE, MATRIX_SIZE, device=device)

    torch.cuda.synchronize()
    start = time.time()
    iterations = 0
    with torch.no_grad():
        while time.time() - start < DURATION_SECONDS:
            a = torch.mm(a, b)
            iterations += 1

    torch.cuda.synchronize()
    elapsed = time.time() - start
    print(f"Completed {iterations} matrix multiplications in {elapsed:.2f} seconds")
    print("Check `nvidia-smi` in another shell to monitor usage while this runs.")


if __name__ == "__main__":
    main()
