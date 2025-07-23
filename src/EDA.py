import os
import numpy as np
import matplotlib.pyplot as plt

def run_eda(args, dataset):
    os.makedirs('doc/fig', exist_ok=True)

    # === Subsample from the Subset ===
    indices = dataset.indices if hasattr(dataset, 'indices') else list(range(len(dataset)))
    rng = np.random.default_rng(seed=0)
    subset_indices = rng.choice(indices, size=args.nsamples, replace=False)

    # === Collect attenuation and tissue labels ===
    all_x = []
    all_y = []
    for idx in subset_indices:
        x, y = dataset.dataset[idx]  # dataset is a Subset of a Dataset
        mu_low  = x[0].flatten().numpy()
        mu_high = x[1].flatten().numpy()
        label   = y.argmax(dim=0).flatten().numpy()  # (512, 512) -> (262144,) labels: 0,1,2

        all_x.append(np.stack([mu_low, mu_high], axis=1))
        all_y.append(label)

    X_all = np.concatenate(all_x, axis=0)
    y_all = np.concatenate(all_y, axis=0)

    # === Scatter plot ===
    plt.figure(figsize=(6, 6))
    for cls, color, label in zip([0, 1, 2], ['gold', 'forestgreen', 'crimson'],
                                 ['Adipose', 'Fibroglandular', 'Calcification']):
        mask = (y_all == cls)
        plt.scatter(X_all[mask, 0], X_all[mask, 1], s=1, alpha=0.2, c=color, label=label)
    plt.xlabel('Raw $\mu_{low}$')
    plt.ylabel('Raw $\mu_{high}$')
    plt.title('Dual-Energy Attenuation by Tissue')
    plt.legend()
    plt.tight_layout()
    plt.savefig('doc/fig/attenuation_scatter.png', dpi=150)
    plt.close()

    # === Histograms ===
    plt.figure(figsize=(6, 8))
    for i, (title, feat_idx) in enumerate([("Low Energy $\mu$", 0), ("High Energy $\mu$", 1)]):
        plt.subplot(2, 1, i + 1)
        for cls, color, label in zip([0, 1, 2], ['gold', 'forestgreen', 'crimson'],
                                     ['Adipose', 'Fibroglandular', 'Calcification']):
            plt.hist(X_all[y_all == cls, feat_idx], bins=100, density=True, alpha=0.6, label=label, color=color)
        plt.title(title)
        plt.xlabel('Attenuation')
        plt.legend()

    plt.tight_layout()
    plt.savefig('doc/fig/attenuation_histograms.png', dpi=150)
    plt.close()

    # === Tissue Percentage per Image ===
    tissue_pct = []
    for i in indices:
        _, y = dataset.dataset[i]  # full (3, 512, 512)
        pct = y.reshape(3, -1).mean(axis=1)
        tissue_pct.append(pct.numpy())
    tissue_pct = np.stack(tissue_pct)

    plt.figure(figsize=(10, 4))
    for i, label in enumerate(['Adipose', 'Fibroglandular', 'Calcification']):
        plt.hist(tissue_pct[:, i] * 100, bins=50, alpha=0.6, label=label)
    plt.xlabel('Tissue Percentage per Image (%)')
    plt.ylabel('Number of Images')
    plt.title('Tissue Type Distribution Across Dataset')
    plt.legend()
    plt.tight_layout()
    plt.savefig('doc/fig/tissue_percentage_distribution.png', dpi=150)
    plt.close()

    avg_pct = tissue_pct.mean(axis=0) * 100
    for lbl, pct in zip(['Adipose', 'Fibroglandular', 'Calcification'], avg_pct):
        print(f"Average {lbl} proportion: {pct:.2f}%")

    # === Calcification Percentage Histogram ===
    plt.figure(figsize=(6, 4))
    plt.hist(tissue_pct[:, 2] * 100, bins=50, alpha=0.7, color='crimson')
    plt.xlabel('Calcification Percentage per Image (%)')
    plt.ylabel('Number of Images')
    plt.title('Calcification Tissue Distribution')
    plt.tight_layout()
    plt.savefig('doc/fig/calcification_distribution.png', dpi=150)
    plt.close()

    print("EDA figures saved to doc/fig/")


# def run_eda(args, ds):
#     os.makedirs('fig', exist_ok=True)

#     # Sample small subset of data
#     rng = np.random.RandomState(0)
#     idxs = rng.choice(len(ds.X_train), size=args.nsamples, replace=False)
#     Xp = ds.X_train[idxs]
#     yp = ds.y_train[idxs]

#     # 1) Scatter plot
#     plt.figure(figsize=(6,6))
#     for cls, color, label in zip([0,1,2], ['gold','forestgreen','crimson'],
#                                  ['Adipose','Fibroglandular','Calcification']):
#         mask = (yp == cls)
#         plt.scatter(Xp[mask,0], Xp[mask,1], s=1, alpha=0.2, c=color, label=label)
#     plt.xlabel('Normalized μ_low')
#     plt.ylabel('Normalized μ_high')
#     plt.title('Dual‑Energy Attenuation by Tissue')
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig('fig/attenuation_scatter.png', dpi=150)
#     plt.close()

#     # 2) Histograms
#     plt.figure(figsize=(12,4))
#     for i, (title, feature_index) in enumerate([('Low Energy μ', 0), ('High Energy μ', 1)]):
#         plt.subplot(1,2,i+1)
#         plt.hist([ds.X_train[ds.y_train==cls,feature_index] for cls in [0,1,2]],
#                  bins=100, density=True,
#                  label=['Adipose','Fibroglandular','Calcification'],
#                  alpha=0.6)
#         plt.title(title)
#         plt.xlabel('Attenuation (standardized)')
#         plt.legend()
#     plt.tight_layout()
#     plt.savefig('fig/attenuation_histograms.png', dpi=150)
#     plt.close()

#     # 3) Tissue Percentage Distribution per Image
#     tissue_pct = []
#     for i in range(len(ds.y)):
#         mask = ds.y[i].numpy()  # shape: (3, 512, 512)
#         pct = mask.reshape(3, -1).mean(axis=1)
#         tissue_pct.append(pct)
#     tissue_pct = np.stack(tissue_pct)  # shape: (N, 3)

#     plt.figure(figsize=(10,4))
#     for i, label in enumerate(['Adipose','Fibroglandular','Calcification']):
#         plt.hist(tissue_pct[:,i]*100, bins=50, alpha=0.6, label=label)
#     plt.xlabel('Tissue Percentage per Image (%)')
#     plt.ylabel('Number of Images')
#     plt.title('Tissue Type Distribution Across Dataset')
#     plt.legend()
#     plt.tight_layout()
#     plt.savefig('fig/tissue_percentage_distribution.png', dpi=150)
#     plt.close()

#     avg_pct = tissue_pct.mean(axis=0) * 100
#     for lbl, pct in zip(['Adipose','Fibroglandular','Calcification'], avg_pct):
#         print(f"Average {lbl} proportion: {pct:.2f}%")

#     # (Add more EDA plots as needed)
#     print("EDA figures saved to fig/")
