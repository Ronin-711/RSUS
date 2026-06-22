import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from mmseg.apis import init_model, inference_model
from ripser import ripser
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from scipy import ndimage
from scipy.optimize import linear_sum_assignment

# --- 视角调节数值接口 ---
ELEVATION = 35  # 俯仰角
AZIMUTH = 300  # 方位角
GT_MASK_PATH = None  # 可选：填入真值 mask 路径后启用距离指标与 Betti error
MAX_POINTS_PER_CLASS = 500
MIN_POINTS_PER_CLASS = 15
TOPK = 10


class FeatureHook:
    def __init__(self): self.feature = None

    def __call__(self, module, input, output): self.feature = output.detach().cpu()


def finite_diagram(diagram):
    if len(diagram) == 0:
        return np.empty((0, 2), dtype=float)
    return diagram[np.isfinite(diagram).all(axis=1)].astype(float)


def persistence_values(diagram):
    diagram = finite_diagram(diagram)
    if len(diagram) == 0:
        return np.array([], dtype=float)
    return np.maximum(diagram[:, 1] - diagram[:, 0], 0.0)


def topk_mean(values, k=TOPK):
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return np.nan
    return np.sort(values)[-min(k, len(values)):].mean()


def persistent_entropy(values):
    values = np.asarray(values, dtype=float)
    values = values[values > 0]
    total = values.sum()
    if total <= 0:
        return np.nan
    probs = values / total
    return float(-(probs * np.log(probs)).sum())


def diagram_summary(diagram, prefix):
    pers = persistence_values(diagram)
    return {
        f'{prefix}_count': len(pers),
        f'{prefix}_mean': float(np.mean(pers)) if len(pers) else np.nan,
        f'{prefix}_max': float(np.max(pers)) if len(pers) else np.nan,
        f'{prefix}_top{TOPK}_mean': float(topk_mean(pers)),
        f'{prefix}_total': float(np.sum(pers)) if len(pers) else 0.0,
        f'{prefix}_entropy': persistent_entropy(pers),
    }


def diagonal_distance(point, ord_value):
    return abs(point[1] - point[0]) / (2.0 if ord_value == np.inf else np.sqrt(2.0))


def augmented_cost_matrix(dgm_a, dgm_b, ord_value):
    dgm_a = finite_diagram(dgm_a)
    dgm_b = finite_diagram(dgm_b)
    n, m = len(dgm_a), len(dgm_b)
    size = n + m
    costs = np.zeros((size, size), dtype=float)

    for i in range(n):
        for j in range(m):
            costs[i, j] = np.linalg.norm(dgm_a[i] - dgm_b[j], ord=ord_value)
        for j in range(n):
            costs[i, m + j] = diagonal_distance(dgm_a[i], ord_value) if i == j else np.inf

    for i in range(m):
        for j in range(m):
            costs[n + i, j] = diagonal_distance(dgm_b[j], ord_value) if i == j else np.inf
        costs[n + i, m:] = 0.0

    return costs


def bottleneck_distance(dgm_a, dgm_b):
    costs = augmented_cost_matrix(dgm_a, dgm_b, np.inf)
    if costs.size == 0:
        return 0.0
    finite_costs = np.unique(costs[np.isfinite(costs)])
    if len(finite_costs) == 0:
        return np.nan

    lo, hi = 0, len(finite_costs) - 1
    answer = finite_costs[-1]
    while lo <= hi:
        mid = (lo + hi) // 2
        threshold = finite_costs[mid]
        feasible_costs = np.where(costs <= threshold, 0.0, 1.0)
        rows, cols = linear_sum_assignment(feasible_costs)
        if feasible_costs[rows, cols].sum() == 0:
            answer = threshold
            hi = mid - 1
        else:
            lo = mid + 1
    return float(answer)


def wasserstein_distance(dgm_a, dgm_b, p=1):
    costs = augmented_cost_matrix(dgm_a, dgm_b, 2)
    if costs.size == 0:
        return 0.0
    large = np.nanmax(costs[np.isfinite(costs)]) + 1.0
    safe_costs = np.where(np.isfinite(costs), costs, large * costs.size)
    rows, cols = linear_sum_assignment(safe_costs ** p)
    return float((safe_costs[rows, cols] ** p).sum() ** (1.0 / p))


def load_mask(path, size_hw):
    if path is None:
        return None
    image = Image.open(path)
    image = image.resize((size_hw[1], size_hw[0]), Image.Resampling.NEAREST)
    return np.array(image).astype(int)


def compute_class_diagrams(feat_flat, mask_flat, classes):
    diagrams = {}
    for cls_id in classes:
        cls_feat = feat_flat[mask_flat == cls_id]
        if len(cls_feat) > MAX_POINTS_PER_CLASS:
            cls_feat = cls_feat[np.random.choice(len(cls_feat), MAX_POINTS_PER_CLASS, replace=False)]
        if len(cls_feat) < MIN_POINTS_PER_CLASS:
            continue
        diagrams[cls_id] = ripser(cls_feat, maxdim=1)['dgms']
    return diagrams


def betti_numbers(mask):
    binary = mask.astype(bool)
    structure = np.ones((3, 3), dtype=int)
    fg_labels, beta0 = ndimage.label(binary, structure=structure)

    bg_labels, bg_count = ndimage.label(~binary, structure=structure)
    if bg_count == 0:
        return int(beta0), 0

    border_labels = set(np.unique(bg_labels[0, :]))
    border_labels.update(np.unique(bg_labels[-1, :]))
    border_labels.update(np.unique(bg_labels[:, 0]))
    border_labels.update(np.unique(bg_labels[:, -1]))
    holes = [label for label in range(1, bg_count + 1) if label not in border_labels]
    return int(beta0), int(len(holes))


def print_metrics_table(rows):
    if not rows:
        print('\n[PH Metrics] No valid class diagrams were computed.')
        return
    df = pd.DataFrame(rows)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        mean_row = df[numeric_cols].mean(numeric_only=True)
        mean_row['Class'] = 'Mean'
        df = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)
    with pd.option_context('display.max_columns', None, 'display.width', 220):
        print('\n[PH Quantitative Metrics]')
        print(df.to_string(index=False, float_format=lambda x: f'{x:.6f}'))


def main():
    # --- 1. 配置与模型加载 ---
    config = CONFIG
    checkpoint = CHECKPOINT
    device = 'cuda:0'
    img_path = IMG_PATH
    gt_mask_path = GT_MASK_PATH

    model = init_model(config, checkpoint, device=device)
    hook = FeatureHook()
    handle = model.decode_head.conv_seg.register_forward_hook(hook)
    result = inference_model(model, img_path)
    feature, _ = hook.feature, handle.remove()

    pred_mask = result.pred_sem_seg.data.cpu()
    B, C, H_f, W_f = feature.shape
    pred_mask_resized = F.interpolate(pred_mask.float().unsqueeze(0), size=(H_f, W_f),
                                      mode='nearest').squeeze().numpy().astype(int)
    feat_flat = feature.squeeze(0).permute(1, 2, 0).reshape(-1, C).numpy()
    mask_flat = pred_mask_resized.flatten()
    unique_classes = [c for c in np.unique(mask_flat) if c != 255]
    gt_mask_resized = load_mask(gt_mask_path, (H_f, W_f))
    gt_flat = gt_mask_resized.flatten() if gt_mask_resized is not None else None
    gt_diagrams = {}
    pred_diagrams = {}

    h0_data, h1_data = [], []

    # --- 配色方案优化：避开黄色 ---
    base_cmap = plt.cm.get_cmap('tab10')
    # 避开 tab10 中的黄色 (索引 8) 和浅色系
    color_indices = [0, 1, 2, 3, 4, 5, 6, 7, 9]
    class_colors = [base_cmap(i) for i in color_indices]

    # --- 2. 拓扑计算 ---
    max_val = 0
    for idx, cls_id in enumerate(unique_classes):
        cls_feat = feat_flat[mask_flat == cls_id]
        if len(cls_feat) > MAX_POINTS_PER_CLASS:
            cls_feat = cls_feat[np.random.choice(len(cls_feat), MAX_POINTS_PER_CLASS, replace=False)]
        if len(cls_feat) < MIN_POINTS_PER_CLASS: continue

        dgms = ripser(cls_feat, maxdim=1)['dgms']
        pred_diagrams[cls_id] = dgms
        h0, h1 = dgms[0], dgms[1]

        h0_finite = h0[np.isfinite(h0[:, 1])][:, 1]
        for val in h0_finite:
            h0_data.append({'Class': f"C{cls_id}", 'Gap_Dist': val, 'Idx': idx})

        if len(h1) > 0:
            h1_persistence = h1[:, 1] - h1[:, 0]
            mask = h1_persistence > np.percentile(h1_persistence, 25)
            for p, pers in zip(h1[mask], h1_persistence[mask]):
                h1_data.append({
                    'Birth': p[0], 'Death': p[1], 'Z': idx,
                    'Persistence': pers, 'Color': class_colors[idx % len(class_colors)]
                })
            max_val = max(max_val, h1.max())

    gt_diagrams = compute_class_diagrams(feat_flat, gt_flat, unique_classes) if gt_flat is not None else {}

    metric_rows = []
    pred_mask_np = pred_mask.squeeze().numpy().astype(int)
    gt_mask_for_betti = load_mask(gt_mask_path, pred_mask_np.shape) if gt_mask_path is not None else None
    for cls_id in unique_classes:
        if cls_id not in pred_diagrams:
            continue
        h0, h1 = pred_diagrams[cls_id]
        row = {'Class': int(cls_id)}
        row.update(diagram_summary(h0, 'H0'))
        row.update(diagram_summary(h1, 'H1'))

        if cls_id in gt_diagrams:
            gt_h0, gt_h1 = gt_diagrams[cls_id]
            row['H0_Bottleneck'] = bottleneck_distance(h0, gt_h0)
            row['H1_Bottleneck'] = bottleneck_distance(h1, gt_h1)
            row['H0_Wasserstein'] = wasserstein_distance(h0, gt_h0)
            row['H1_Wasserstein'] = wasserstein_distance(h1, gt_h1)
        else:
            row['H0_Bottleneck'] = np.nan
            row['H1_Bottleneck'] = np.nan
            row['H0_Wasserstein'] = np.nan
            row['H1_Wasserstein'] = np.nan

        if gt_mask_for_betti is not None:
            pred_b0, pred_b1 = betti_numbers(pred_mask_np == cls_id)
            gt_b0, gt_b1 = betti_numbers(gt_mask_for_betti == cls_id)
            row['Betti0_Error'] = abs(pred_b0 - gt_b0)
            row['Betti1_Error'] = abs(pred_b1 - gt_b1)
        else:
            row['Betti0_Error'] = np.nan
            row['Betti1_Error'] = np.nan
        metric_rows.append(row)

    print_metrics_table(metric_rows)

    # --- 3. 图 1：H0 二维分布 ---
    plt.figure(figsize=(10, 4))
    df_h0 = pd.DataFrame(h0_data)
    for idx, cls_name in enumerate(df_h0['Class'].unique()):
        sub = df_h0[df_h0['Class'] == cls_name]
        plt.scatter(sub['Gap_Dist'], np.full(len(sub), idx),
                    alpha=0.7, color=class_colors[idx % len(class_colors)], s=30, edgecolors='white', linewidth=0.5)
    plt.title("H0: Feature Gap Analysis (Remote Sensing Interruption)")
    plt.xlabel("Merging Distance")
    plt.yticks(range(len(unique_classes)), [f"Class {c}" for c in unique_classes])
    plt.grid(True, axis='x', linestyle='--', alpha=0.3)

    # --- 4. 图 2：H1 三维结构稳定性图 (无黄色版) ---
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    lim = max_val * 1.1

    for idx, cls_id in enumerate(unique_classes):
        # A. 参考虚线 (加深的中灰色)
        ax.plot([0, lim], [0, lim], [idx, idx], color='#444444', linestyle='--', linewidth=1.2, alpha=0.5)

        cls_h1 = [d for d in h1_data if d['Z'] == idx]
        if not cls_h1: continue

        births = np.array([d['Birth'] for d in cls_h1])
        deaths = np.array([d['Death'] for d in cls_h1])
        pers = np.array([d['Persistence'] for d in cls_h1])
        c = class_colors[idx % len(class_colors)]

        # B. 强化后的散点样式
        # 增加边框粗细，提升 3D 层级感
        ax.scatter(births, deaths, [idx] * len(births),
                   color=c,
                   s=pers * 100 + 30,
                   edgecolors='#222222',
                   linewidth=1.2,
                   alpha=0.9)

        # C. 投影点线 (与散点同色，增强追踪性)
        for b, d in zip(births, deaths):
            ax.plot([b, b], [d, b], [idx, idx], color=c, linestyle=':', linewidth=1.8, alpha=0.8)

    # 5. UI 细节调整
    ax.view_init(elev=ELEVATION, azim=AZIMUTH)
    ax.set_title("H1 Persistence: Structural Stability Analysis", pad=30, fontsize=15, fontweight='bold')
    ax.set_xlabel('Formation Scale (Birth)', fontsize=12, labelpad=12)
    ax.set_ylabel('Stability Scale (Death)', fontsize=12, labelpad=12)
    ax.set_zlabel('Semantic Class Index', fontsize=12, labelpad=12)

    ax.set_zticks(range(len(unique_classes)))
    ax.set_zticklabels([f"Class {c}" for c in unique_classes])

    # 彻底去除背景杂色，突出数据
    ax.xaxis.pane.fill = False;
    ax.yaxis.pane.fill = False;
    ax.zaxis.pane.fill = False
    ax.set_xlim(0, lim);
    ax.set_ylim(0, lim)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
