import os
import numpy as np
import matplotlib.pyplot as plt

def run_eda(args, ds):
    os.makedirs('fig', exist_ok=True)

    # Sample small subset of data
    rng = np.random.RandomState(0)
    idxs = rng.choice(len(ds.X_train), size=args.nsamples, replace=False)
    Xp = ds.X_train[idxs]
    yp = ds.y_train[idxs]

    # 1) Scatter plot
    plt.figure(figsize=(6,6))
    for cls, color, label in zip([0,1,2], ['gold','forestgreen','crimson'],
                                 ['Adipose','Fibroglandular','Calcification']):
        mask = (yp == cls)
        plt.scatter(Xp[mask,0], Xp[mask,1], s=1, alpha=0.2, c=color, label=label)
    plt.xlabel('Normalized μ_low')
    plt.ylabel('Normalized μ_high')
    plt.title('Dual‑Energy Attenuation by Tissue')
    plt.legend()
    plt.tight_layout()
    plt.savefig('fig/attenuation_scatter.png', dpi=150)
    plt.close()

    # 2) Histograms
    plt.figure(figsize=(12,4))
    for i, (title, feature_index) in enumerate([('Low Energy μ', 0), ('High Energy μ', 1)]):
        plt.subplot(1,2,i+1)
        plt.hist([ds.X_train[ds.y_train==cls,feature_index] for cls in [0,1,2]],
                 bins=100, density=True,
                 label=['Adipose','Fibroglandular','Calcification'],
                 alpha=0.6)
        plt.title(title)
        plt.xlabel('Attenuation (standardized)')
        plt.legend()
    plt.tight_layout()
    plt.savefig('fig/attenuation_histograms.png', dpi=150)
    plt.close()

    # (Add more EDA plots as needed)
    print("EDA figures saved to fig/")
