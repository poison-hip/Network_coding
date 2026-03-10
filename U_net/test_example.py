import torch

from unet import UNet


def run_test_example():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = UNet(in_channels=3, num_classes=2, bilinear=True, base_c=64).to(device)
    model.train()

    # 模拟一批 RGB 图片和对应分割标签
    x = torch.randn(2, 3, 256, 256, device=device)
    target = torch.randint(0, 2, (2, 256, 256), device=device)

    logits = model(x)
    print("logits shape:", logits.shape)  # [N, num_classes, H, W]

    # 前向输出转预测 mask
    probs = torch.softmax(logits, dim=1)
    pred_mask = torch.argmax(probs, dim=1)
    print("pred mask shape:", pred_mask.shape)  # [N, H, W]

    # 一次反向传播，验证训练流程可跑通
    loss_fn = torch.nn.CrossEntropyLoss()
    loss = loss_fn(logits, target)
    loss.backward()
    print("loss:", loss.detach().item())


if __name__ == "__main__":
    run_test_example()
