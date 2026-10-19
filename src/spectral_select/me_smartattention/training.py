import gc
import random
import numpy as np
import torch
import torch.nn as nn

from tqdm.auto import tqdm

from torch.utils.data import (
    Dataset,
    DataLoader,
    WeightedRandomSampler
)

from sklearn.metrics import (
    accuracy_score,
    f1_score
)

from .smart_attention import MESmartAttention
from .baseline import specimen_vote_accuracy


# ============================================================
# DATASET
# ============================================================

class MEDataset(Dataset):

    def __init__(
        self,
        X,
        y,
        groups
    ):
        self.X = torch.from_numpy(
            X.astype(np.float32)
        )

        self.y = torch.from_numpy(
            y.astype(np.int64)
        )

        self.groups = torch.from_numpy(
            groups.astype(np.int64)
        )

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (
            self.X[i],
            self.y[i],
            self.groups[i]
        )


# ============================================================
# SEED
# ============================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# TRAIN-ONLY STANDARDIZATION
# ============================================================

def standardize_eem(
    X_eem,
    eem_mask,
    train_idx,
    test_idx
):

    E, M = eem_mask.shape

    mean = np.zeros(
        (E, M),
        dtype=np.float32
    )

    std = np.ones(
        (E, M),
        dtype=np.float32
    )

    train_raw = X_eem[
        train_idx
    ]

    mean[eem_mask] = train_raw[
        :, eem_mask
    ].mean(axis=0)

    std[eem_mask] = train_raw[
        :, eem_mask
    ].std(axis=0)

    std[
        std < 1e-8
    ] = 1.0


    def transform(X):

        Z = (
            (
                X
                - mean[None]
            )
            / std[None]
        ).astype(
            np.float32
        )

        # Physically nonexistent EEM measurements
        # remain zero.
        Z[
            :, ~eem_mask
        ] = 0.0

        return Z


    return (
        transform(
            X_eem[train_idx]
        ),

        transform(
            X_eem[test_idx]
        )
    )


# ============================================================
# SPECIMEN-BALANCED TRAIN LOADER
# ============================================================

def make_train_loader(
    X,
    y,
    groups,

    batch_size=256,
    samples_per_specimen=2500,

    seed=42,
    num_workers=0
):

    assert np.all(
        groups >= 0
    )

    dataset = MEDataset(
        X,
        y,
        groups
    )

    specimens = np.unique(
        groups
    )

    weights = np.zeros(
        len(groups),
        dtype=np.float64
    )

    for sid in specimens:

        m = (
            groups == sid
        )

        weights[m] = (
            1.0 / m.sum()
        )


    generator = torch.Generator()

    generator.manual_seed(
        seed
    )


    sampler = WeightedRandomSampler(
        torch.tensor(
            weights,
            dtype=torch.double
        ),

        num_samples=(
            samples_per_specimen
            * len(specimens)
        ),

        replacement=True,

        generator=generator
    )


    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers
    )


# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================

def run_smart_attention(
    data,
    splits,

    name="ME SmartAttention",

    # ---------------- MODEL ----------------

    dim=16,
    depth=3,
    heads=2,
    dim_head=8,
    mlp_dim=4,

    dropout=0.10,
    embedding_dropout=0.10,

    edge_mode="zero",

    # ---------------- TRAINING ----------------

    epochs=40,
    batch_size=256,
    samples_per_specimen=2500,

    lr=5e-4,
    weight_decay=0.0,

    # ---------------- SCHEDULER ----------------

    scheduler_gamma=0.9,
    scheduler_steps=10,

    # ---------------- MISC ----------------

    seed=42,
    device=None,
    num_workers=0
):

    X_eem = data[
        "X_eem"
    ]

    eem_mask = data[
        "eem_mask"
    ]

    y = data[
        "y"
    ]

    groups = data[
        "groups"
    ]


    assert np.all(
        groups >= 0
    ), (
        "Invalid specimen IDs found. "
        "Reload the dataset using corrected data.py."
    )


    if edge_mode not in {
        "circular",
        "zero",
        "discard"
    }:
        raise ValueError(
            "edge_mode must be "
            "'circular', 'zero', or 'discard'"
        )


    n_excitations = (
        X_eem.shape[1]
    )

    n_emissions = (
        X_eem.shape[2]
    )

    n_classes = len(
        np.unique(y)
    )


    if device is None:

        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            device = torch.device("mps")
        else:
            device = torch.device("cpu")


    print(
        "\n"
        + "=" * 70
    )

    print(name)

    print(
        "=" * 70
    )


    all_test_idx = []
    all_predictions = []

    fold_results = []


    # ========================================================
    # FOLDS
    # ========================================================

    for fold, (
        tr,
        te
    ) in enumerate(
        splits,
        1
    ):

        print(
            f"\nFold {fold}"
        )

        print(
            "Train specimens:",
            sorted(
                np.unique(
                    groups[tr]
                ).tolist()
            )
        )

        print(
            "Test specimens:",
            sorted(
                np.unique(
                    groups[te]
                ).tolist()
            )
        )


        # ====================================================
        # TRAIN-ONLY STANDARDIZATION
        # ====================================================

        X_train, X_test = standardize_eem(
            X_eem,
            eem_mask,
            tr,
            te
        )

        y_train = y[
            tr
        ]

        y_test = y[
            te
        ]

        g_train = groups[
            tr
        ]


        # ====================================================
        # SEED
        # ====================================================

        fold_seed = (
            seed + fold
        )

        set_seed(
            fold_seed
        )


        # ====================================================
        # TRAIN LOADER
        # ====================================================

        train_loader = make_train_loader(
            X_train,
            y_train,
            g_train,

            batch_size=
                batch_size,

            samples_per_specimen=
                samples_per_specimen,

            seed=
                fold_seed,

            num_workers=
                num_workers
        )


        # ====================================================
        # MODEL
        # ====================================================

        model = MESmartAttention(
            n_excitations=
                n_excitations,

            n_emissions=
                n_emissions,

            n_classes=
                n_classes,

            dim=
                dim,

            depth=
                depth,

            heads=
                heads,

            dim_head=
                dim_head,

            mlp_dim=
                mlp_dim,

            dropout=
                dropout,

            embedding_dropout=
                embedding_dropout,

            edge_mode=
                edge_mode
        ).to(
            device
        )


        # ====================================================
        # SHOW CONFIG
        # ====================================================

        if fold == 1:

            params = sum(
                p.numel()
                for p in model.parameters()
                if p.requires_grad
            )

            if edge_mode == "discard":
                emission_tokens = (
                    n_emissions - 2
                )
            else:
                emission_tokens = (
                    n_emissions
                )

            print(
                "\nModel configuration"
            )

            print(
                f"Parameters: {params:,}"
            )

            print(
                f"DIM: {dim}"
            )

            print(
                f"Depth: {depth}"
            )

            print(
                f"Heads: {heads}"
            )

            print(
                f"Head dimension: "
                f"{dim_head}"
            )

            print(
                f"FFN dimension: "
                f"{mlp_dim}"
            )

            print(
                f"Dropout: "
                f"{dropout}"
            )

            print(
                f"Embedding dropout: "
                f"{embedding_dropout}"
            )

            print(
                f"Edge mode: "
                f"{edge_mode}"
            )

            print(
                f"Emission tokens: "
                f"{emission_tokens}"
            )

            print(
                f"Epochs: "
                f"{epochs}"
            )

            print(
                f"Learning rate: "
                f"{lr}"
            )

            print(
                f"Weight decay: "
                f"{weight_decay}"
            )

            print(
                f"Batch size: "
                f"{batch_size}"
            )

            print(
                f"Samples/specimen: "
                f"{samples_per_specimen}"
            )


        # ====================================================
        # OPTIMIZER
        # ====================================================

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=
                weight_decay
        )


        # ====================================================
        # SCHEDULER
        # ====================================================

        step_size = max(
            1,
            epochs
            // scheduler_steps
        )

        scheduler = (
            torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=
                    step_size,
                gamma=
                    scheduler_gamma
            )
        )


        criterion = (
            nn.CrossEntropyLoss()
        )


        # ====================================================
        # TRAIN
        # ====================================================

        bar = tqdm(
            range(
                1,
                epochs + 1
            ),
            desc=f"Fold {fold}"
        )


        for epoch in bar:

            model.train()

            loss_sum = 0.0
            correct = 0
            total = 0


            for xb, yb, _ in train_loader:

                xb = xb.to(
                    device
                )

                yb = yb.to(
                    device
                )


                optimizer.zero_grad(
                    set_to_none=True
                )


                logits = model(
                    xb
                )


                loss = criterion(
                    logits,
                    yb
                )


                loss.backward()


                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    1.0
                )


                optimizer.step()


                loss_sum += (
                    loss.item()
                    * len(yb)
                )


                correct += (
                    logits.argmax(1)
                    == yb
                ).sum().item()


                total += len(yb)


            scheduler.step()


            bar.set_postfix(
                loss=
                    f"{loss_sum / total:.3f}",

                train_acc=
                    f"{correct / total:.3f}"
            )


        # ====================================================
        # TEST
        # ====================================================

        model.eval()


        test_dataset = MEDataset(
            X_test,
            y_test,
            groups[te]
        )


        test_loader = DataLoader(
            test_dataset,
            batch_size=1024,
            shuffle=False,
            num_workers=num_workers
        )


        predictions = []


        with torch.no_grad():

            for xb, _, _ in test_loader:

                xb = xb.to(
                    device
                )


                pred = (
                    model(xb)
                    .argmax(1)
                    .cpu()
                    .numpy()
                )


                predictions.append(
                    pred
                )


        pred = np.concatenate(
            predictions
        )


        # ====================================================
        # FOLD METRICS
        # ====================================================

        pixel_acc = accuracy_score(
            y_test,
            pred
        )


        macro_f1 = f1_score(
            y_test,
            pred,
            average="macro",
            labels=np.arange(n_classes),
            zero_division=0
        )


        sc, sn = specimen_vote_accuracy(
            te,
            pred,
            y,
            groups
        )


        print(
            f"Fold {fold}: "
            f"pixel {pixel_acc:.2%} | "
            f"F1 {macro_f1:.4f} | "
            f"specimen "
            f"{sc}/{sn} "
            f"({sc/sn:.2%})"
        )


        fold_results.append({
            "fold":
                fold,

            "pixel_accuracy":
                pixel_acc,

            "macro_f1":
                macro_f1,

            "specimen_correct":
                sc,

            "specimen_total":
                sn
        })


        all_test_idx.append(
            te
        )

        all_predictions.append(
            pred
        )


        # ====================================================
        # CLEANUP
        # ====================================================

        del (
            model,
            X_train,
            X_test,
            train_loader,
            test_loader
        )

        gc.collect()


        if torch.cuda.is_available():
            torch.cuda.empty_cache()


    # ========================================================
    # OVERALL METRICS
    # ========================================================

    all_test_idx = np.concatenate(
        all_test_idx
    )

    all_predictions = np.concatenate(
        all_predictions
    )


    overall_acc = accuracy_score(
        y[
            all_test_idx
        ],
        all_predictions
    )


    overall_f1 = f1_score(
        y[
            all_test_idx
        ],
        all_predictions,
        average="macro",
        labels=np.arange(n_classes),
        zero_division=0
    )


    sc, sn = specimen_vote_accuracy(
        all_test_idx,
        all_predictions,
        y,
        groups
    )


    print(
        "\nOverall pixel accuracy: "
        f"{overall_acc:.2%}"
    )

    print(
        "Macro F1: "
        f"{overall_f1:.4f}"
    )

    print(
        "Specimen majority-vote accuracy: "
        f"{sc}/{sn} "
        f"({sc/sn:.2%})"
    )


    return {
        "pixel_accuracy":
            overall_acc,

        "macro_f1":
            overall_f1,

        "specimen_correct":
            sc,

        "specimen_total":
            sn,

        "folds":
            fold_results,

        "predictions":
            all_predictions,

        "test_indices":
            all_test_idx
    }