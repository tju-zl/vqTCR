import umap
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import pandas as pd

def plot_codebook_umap(model, show_roots=True, show_leaves=True,
                       n_neighbors=10, palette="tab20", plot_on="leaf",
                       text_bias=0.5, l_dim=None):
    """
    Args:
        model: MrTCR model (model.clone_pty)
        show_roots: plot root centroid
        show_leaves: plot active leaves
        n_neighbors: UMAP neighborhood size
        palette:  "tab10", "tab20", "Spectral", 'viridis
        plot_on: "all" or "leaf"
        text_bis: labels position bias
    """
    roots, leaves, idx = model.clone_pty.get_protype()

    leaves = leaves.reshape(-1, l_dim) # flatten

    if not show_roots and not show_leaves:
        raise ValueError("At least show_roots or show_leaves = True")

    # --- Fit UMAP ---
    if plot_on == "all":
        all_codes = []
        if show_roots:
            all_codes.append(roots)
        if show_leaves:
            all_codes.append(leaves)
        all_codes = np.vstack(all_codes)

        umap_model = umap.UMAP(n_components=2, n_neighbors=n_neighbors, random_state=42)
        all_umap = umap_model.fit_transform(all_codes)

        pos = 0
        roots_umap, codes_umap = None, None
        if show_roots:
            roots_umap = all_umap[pos:pos + roots.shape[0]]
            pos += roots.shape[0]
        if show_leaves:
            codes_umap = all_umap[pos:pos + leaves.shape[0]]

    elif plot_on == "leaf":
        umap_model = umap.UMAP(n_components=2, n_neighbors=n_neighbors, random_state=42)
        codes_umap = umap_model.fit_transform(leaves) if show_leaves else None
        roots_umap = umap_model.transform(roots) if show_roots else None

    else:
        raise ValueError("fit_on is 'all' or 'leaf'")

    # --- plotting ---
    plt.figure(figsize=(7, 5))
    if show_leaves and codes_umap is not None:
        sns.scatterplot(
            x=codes_umap[:, 0], y=codes_umap[:, 1],
            hue=idx[:, 0],  # root id 给叶子上色
            palette=palette, s=30, alpha=0.7, legend=False, zorder=1
        )
    if show_roots and roots_umap is not None:
        plt.scatter(
            roots_umap[:, 0], roots_umap[:, 1],
            c=list(range(roots.shape[0])), cmap=palette,
            s=150, marker="X", edgecolor="black", zorder=2, label="root"
        )
        for i, (x, y) in enumerate(roots_umap):
            plt.text(
                x+text_bias, y+text_bias, f"R{i}",  # R0, R1, R2...
                fontsize=8, ha="center", va="center",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", boxstyle="round"),
                zorder=3
            )

    plt.title(f"UMAP of Corse and Subtle Clonotypes (plot_on={plot_on})")
    plt.xlabel("UMAP 1"); plt.ylabel("UMAP 2")
    plt.show()




















#===================
def plot_cells_by_prototype(adata, key="pty_corse", palette="tab20", method="umap"):
    """
    use adata.obs to plot prototype assignment in cells

    Args:
        adata: AnnData ( .obsm['htcr'] or .obsm['hgex'])
        key: str, adata.obs column name: prototype assignment,  "pty_corse" or "pty_subtle"
        palette: "tab10", "tab20", "Spectral", 'viridis
        method: "umap" or "tsne"
    """
    if "htcr" not in adata.obsm:
        raise ValueError("use MrTCR.get_latent() get htcr latent")

    X = adata.obsm["htcr"]

    if method == "umap":
        reducer = umap.UMAP(n_components=2, n_neighbors=15, random_state=42)
        coords = reducer.fit_transform(X)
    elif method == "tsne":
        from sklearn.manifold import TSNE
        coords = TSNE(n_components=2, random_state=42).fit_transform(X)
    else:
        raise ValueError("method must be 'umap' or 'tsne'")

    plt.figure(figsize=(7, 5))
    sns.scatterplot(
        x=coords[:, 0], y=coords[:, 1],
        hue=adata.obs[key].astype(str),
        palette=palette, s=15, alpha=0.8, linewidth=0
    )
    plt.title(f"Cells colored by {key}")
    plt.xlabel(f"{method.upper()} 1"); plt.ylabel(f"{method.upper()} 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title=key)
    plt.tight_layout()
    plt.show()


def plot_prototype_annotation(adata, proto_key="pty_corse", palette="tab20", top_n=10):
    """
    plot prototype distribution, barplot
    Args:
        adata: AnnData,  adata.uns with {proto_key}_summary
        proto_key: "pty_corse" or "pty_subtle"
        palette: "tab10", "tab20", "Spectral", 'viridis
        top_n: top n prototype labels
    """
    if f"{proto_key}_summary" not in adata.uns:
        raise ValueError(f"call MrTCR.summarize_prototype_labels to save {proto_key}_summary")

    proto_summary = adata.uns[f"{proto_key}_summary"]

    fig, axes = plt.subplots(len(proto_summary), 1, figsize=(7, 3 * len(proto_summary)), sharex=True)
    if len(proto_summary) == 1:
        axes = [axes]

    for ax, (pid, counts) in zip(axes, proto_summary.items()):
        items = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
        labels, values = zip(*items)
        sns.barplot(x=list(values), y=list(labels), palette=palette, ax=ax)
        ax.set_title(f"{proto_key} {pid} - top {top_n} labels")
        ax.set_xlabel("Count"); ax.set_ylabel("Label")

    plt.tight_layout()
    plt.show()


def plot_clonotype_on_prototype(adata, proto_key="pty_corse", clonotype_key="clonotype", palette="tab20"):
    """
    plot clonotype distribution on prototype assignment
    Args:
        adata: AnnData with adata.obs[proto_key] and adata.obs[clonotype_key]
        proto_key: "pty_corse" or "pty_subtle"
        clonotype_key: adata.obs clonotype key
        palette: "tab10", "tab20", "Spectral", 'viridis
    """

    if proto_key not in adata.obs or clonotype_key not in adata.obs:
        raise ValueError(f"adata.obs 里缺少 {proto_key} 或 {clonotype_key}")

    cross_tab = pd.crosstab(adata.obs[proto_key], adata.obs[clonotype_key])

    plt.figure(figsize=(10, 6))
    sns.heatmap(cross_tab, cmap="Blues", cbar=True)
    plt.title(f"Clonotype distribution across {proto_key}")
    plt.xlabel("Clonotype")
    plt.ylabel(proto_key)
    plt.tight_layout()
    plt.show()

    return cross_tab


def plot_clonotype_in_latent(adata, clonotype_key="clonotype", method="umap", 
                             palette="tab20", top_n=50):
    """
    group visualization of latent space (htcr) with clonotype
    Args:
        adata: AnnData with adata.obsm['htcr'] and adata.obs[clonotype_key]
        clonotype_key: adata.obs clonotype key
        method: "umap" or "tsne"
        palette: "tab10", "tab20", "Spectral", 'viridis
    """
    from sklearn.manifold import TSNE
    import umap
    import seaborn as sns
    import matplotlib.pyplot as plt

    if "htcr" not in adata.obsm:
        raise ValueError("call MrTCR.get_latent() to get htcr representation")

    X = adata.obsm["htcr"]

    if method == "umap":
        reducer = umap.UMAP(n_components=2, n_neighbors=15, random_state=42)
        coords = reducer.fit_transform(X)
    elif method == "tsne":
        coords = TSNE(n_components=2, random_state=42).fit_transform(X)
    else:
        raise ValueError("method must be 'umap' or 'tsne'")

    plt.figure(figsize=(7, 5))
    sns.scatterplot(
        x=coords[:, 0], y=coords[:, 1],
        hue=adata.obs[clonotype_key].astype(str),
        palette=palette, s=15, alpha=0.8, linewidth=0
    )
    plt.title(f"{method.upper()} of latent htcr colored by {clonotype_key}")
    plt.xlabel(f"{method.upper()} 1"); plt.ylabel(f"{method.upper()} 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title=clonotype_key)
    plt.tight_layout()
    plt.show()
