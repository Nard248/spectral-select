import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score
)


def specimen_vote_accuracy(
    test_idx,
    predictions,
    y,
    groups
):
    test_groups = groups[
        test_idx
    ]

    assert np.all(
        test_groups >= 0
    )


    specimens = np.unique(
        test_groups
    )

    n_classes = len(
        np.unique(y)
    )

    correct = 0


    for sid in specimens:

        m = (
            test_groups
            == sid
        )


        true_class = int(
            y[
                test_idx[m]
            ][0]
        )


        predicted_class = int(
            np.bincount(
                predictions[m],
                minlength=n_classes
            ).argmax()
        )


        correct += (
            true_class
            == predicted_class
        )


    return (
        correct,
        len(specimens)
    )


def run_logistic_regression(
    X,
    y,
    groups,
    splits,
    name="Logistic Regression",
    seed=42
):
    assert np.all(
        groups >= 0
    )


    print(
        "\n"
        + "=" * 70
    )

    print(
        name
    )

    print(
        "=" * 70
    )


    all_test_idx = []
    all_predictions = []

    fold_results = []


    for fold, (
        tr,
        te
    ) in enumerate(
        splits,
        1
    ):

        # ====================================================
        # TRAIN-ONLY STANDARDIZATION
        # ====================================================

        scaler = StandardScaler()


        X_train = scaler.fit_transform(
            X[tr]
        )

        X_test = scaler.transform(
            X[te]
        )


        y_train = y[
            tr
        ]

        y_test = y[
            te
        ]


        train_groups = groups[
            tr
        ]


        # ====================================================
        # EQUAL SPECIMEN CONTRIBUTION
        # ====================================================

        weights = np.zeros(
            len(tr),
            dtype=np.float64
        )


        for sid in np.unique(
            train_groups
        ):

            m = (
                train_groups
                == sid
            )

            weights[
                m
            ] = (
                1.0
                / m.sum()
            )


        # ====================================================
        # MODEL
        # ====================================================

        model = LogisticRegression(
            C=1.0,
            penalty="l2",
            solver="lbfgs",
            max_iter=2000,
            random_state=seed
        )


        model.fit(
            X_train,
            y_train,
            sample_weight=weights
        )


        pred = model.predict(
            X_test
        )


        # ====================================================
        # METRICS
        # ====================================================

        pixel_acc = accuracy_score(
            y_test,
            pred
        )


        macro_f1 = f1_score(
            y_test,
            pred,
            average="macro",
            labels=np.arange(len(np.unique(y))),
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
                sn,
        })


        all_test_idx.append(
            te
        )

        all_predictions.append(
            pred
        )


    # ========================================================
    # OVERALL
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
        labels=np.arange(len(np.unique(y))),
        zero_division=0
    )


    sc, sn = specimen_vote_accuracy(
        all_test_idx,
        all_predictions,
        y,
        groups
    )


    print(
        f"\nOverall pixel accuracy: "
        f"{overall_acc:.2%}"
    )

    print(
        f"Macro F1: "
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
            all_test_idx,
    }