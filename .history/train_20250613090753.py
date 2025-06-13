import os
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# 设置 DFormer 路径（若 mmseg 不在 site-packages 中）
sys.path.append(os.path.join(os.path.dirname(__file__), "DFormer"))

from config.config import C as config
from models.dformer_backbone import build_dformer, load_pretrained_weights
from models.multipoint_model import DFormerMultiPoint, freeze_encoder, unfreeze_encoder
from data.dataset import MultiPointDataset
from losses.combined_loss import CombinedLoss
from utils.metrics import calculate_pck, calculate_oks
from utils.visualization import visualize_predictions

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train():
    print(f"Using device: {DEVICE}")

    # 构建主干并加载预训练
    base_model = build_dformer(config)
    base_model = load_pretrained_weights(base_model, config["pretrained_path"]).to(DEVICE)

    model = DFormerMultiPoint(
        base_model,
        num_points=config["num_points"],
        heatmap_size=config["heatmap_size"]
    ).to(DEVICE)

    freeze_encoder(model)

    # 损失函数
    criterion = CombinedLoss(
        heatmap_weight=config["heatmap_weight"],
        coord_weight=config["coord_weight"],
        image_size=config["image_size"]
    )

    # 优化器
    optimizer = optim.AdamW([
        {"params": model.encoder.parameters(), "lr": config["lr_encoder"]},
        {"params": model.heatmap_head.parameters(), "lr": config["lr_head"]}
    ], weight_decay=config["weight_decay"])

    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config["scheduler_T_max"],
        eta_min=config["scheduler_eta_min"]
    )

    # 数据集
    train_set = MultiPointDataset(
        config["train_csv"], config["rgb_dir"], config["depth_dir"],
        config, is_train=True
    )
    val_set = MultiPointDataset(
        config["val_csv"], config["rgb_dir"], config["depth_dir"],
        config, is_train=False
    )

    train_loader = DataLoader(
        train_set,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        pin_memory=True
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["val_num_workers"],
        pin_memory=True
    )

    os.makedirs(config["visual_output_dir"], exist_ok=True)
    os.makedirs(config["checkpoint_dir"], exist_ok=True)

    best_val_metric = -float("inf") if config["save_best_by"] in ["pck", "oks"] else float("inf")

    for epoch in range(config["total_epochs"]):
        if epoch == config["freeze_epochs"]:
            unfreeze_encoder(model)

        model.train()
        total_loss = 0
        batch_count = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]"):
            rgb = batch["rgb"].to(DEVICE)
            depth = batch["depth"].to(DEVICE)
            target = {
                "heatmaps": batch["heatmap"].to(DEVICE),
                "coords": batch["coords"].to(DEVICE)
            }

            optimizer.zero_grad()
            output = model(rgb, depth)
            loss, _ = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            batch_size = rgb.size(0)
            total_loss += loss.item() * batch_size
            batch_count += batch_size
        val_pck = 0
        val_oks = 0
        # 验证阶段
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]")):
                rgb = batch["rgb"].to(DEVICE)
                depth = batch["depth"].to(DEVICE)
                target = {
                    "heatmaps": batch["heatmap"].to(DEVICE),
                    "coords": batch["coords"].to(DEVICE)
                }

                output = model(rgb, depth)
                pck = calculate_pck(output["coords"], target["coords"], threshold=0.05)
                oks = calculate_oks(output["coords"], target["coords"], image_size=config["image_size"])
                val_pck += pck * rgb.size(0)
                val_oks += oks * rgb.size(0)

                if i == 0 and epoch % config["visualize_every_n_epoch"] == 0:
                    visualize_predictions(
                        batch["original_rgb"][0],
                        batch["coords"][0].view(-1, 2),
                        output["coords"][0].view(-1, 2),
                        output["heatmaps"][0],
                        save_path=os.path.join(config["visual_output_dir"], f"epoch_{epoch+1}.png")
                    )

        scheduler.step()
        avg_train_loss = total_loss / batch_count
        val_pck /= len(val_set)
        val_oks /= len(val_set)
        print(f"\nEpoch {epoch+1}/{config['total_epochs']} - "
              f"Loss: {avg_train_loss:.4f}, "
              f"Val PCK: {val_pck:.4f}, "
              f"Val OKS: {val_oks:.4f}")
        print("-" * 50)

        # 保存最佳模型
        if (
            (config["save_best_by"] == "pck" and val_pck > best_val_metric) or
            (config["save_best_by"] == "oks" and val_oks > best_val_metric) or
            (config["save_best_by"] == "loss" and total_loss < best_val_metric)
        ):
            best_val_metric = {
                "pck": val_pck,
                "oks": val_oks,
                "loss": total_loss
            }[config["save_best_by"]]
            
            torch.save(model.state_dict(), config["best_model_path"])
            print(f"✅ Saved best model by {config['save_best_by']} = {best_val_metric:.4f}")

if __name__ == "__main__":
    train()
