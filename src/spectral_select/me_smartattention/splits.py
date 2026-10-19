import numpy as np

def _validate_dataset(
    y,
    groups
):
    assert len(y) == len(groups)

    assert np.all(
        groups >= 0
    ), (
        "Invalid specimen IDs found. "
        "Artifact pixels should have "
        "been removed in data.py."
    )

    for sid in np.unique(
        groups
    ):
        assert (
            np.unique(
                y[
                    groups == sid
                ]
            ).size
            == 1
        )


def make_specimen_splits(
    X,
    y,
    groups,
    seed=42
):
    """
    Whole-specimen CV.

    Every specimen belongs entirely to
    either train or test within each fold.
    """

    _validate_dataset(
        y,
        groups
    )


    specimens_per_class = [
        len(
            np.unique(
                groups[
                    y == c
                ]
            )
        )
        for c in np.unique(y)
    ]


    n_folds = min(
        specimens_per_class
    )


    # Class-covering whole-specimen folds: shuffle the specimens of each
    # class (seeded) and deal them round-robin into n_folds folds, so
    # every test fold contains >= 1 specimen of every class.
    # (StratifiedGroupKFold does not guarantee this with few groups.)
    rng = np.random.default_rng(
        seed
    )

    classes = np.unique(
        y
    )

    fold_specimens = [
        []
        for _ in range(n_folds)
    ]

    for c in classes:

        class_specimens = np.unique(
            groups[
                y == c
            ]
        )

        rng.shuffle(
            class_specimens
        )

        for i, sid in enumerate(
            class_specimens
        ):
            fold_specimens[
                i % n_folds
            ].append(
                int(sid)
            )

    splits = []

    for specimens in fold_specimens:

        test_mask = np.isin(
            groups,
            specimens
        )

        splits.append(
            (
                np.flatnonzero(
                    ~test_mask
                ),
                np.flatnonzero(
                    test_mask
                )
            )
        )


    for tr, te in splits:

        assert set(
            np.unique(y[te])
        ) == set(
            classes
        ), (
            "A test fold is missing a class"
        )


    # Safety
    for tr, te in splits:

        assert set(
            groups[tr]
        ).isdisjoint(
            set(
                groups[te]
            )
        )


    return splits


def make_pixel_splits(
    y,
    groups,
    n_folds=3,
    seed=42
):
    """
    Balanced random pixel CV.

    Every specimen may appear in both
    training and testing.

    Test pixel counts are equal across classes.
    """

    _validate_dataset(
        y,
        groups
    )


    rng = np.random.default_rng(
        seed
    )

    classes = np.unique(
        y
    )

    all_idx = np.arange(
        len(y)
    )


    min_class_pixels = min(
        np.sum(
            y == c
        )
        for c in classes
    )


    n_test_per_class = (
        min_class_pixels
        // n_folds
    )


    class_indices = {}


    for c in classes:

        idx = np.flatnonzero(
            y == c
        )

        rng.shuffle(
            idx
        )

        class_indices[
            c
        ] = idx


    splits = []


    for fold in range(
        n_folds
    ):

        start = (
            fold
            * n_test_per_class
        )

        end = (
            start
            + n_test_per_class
        )


        test_idx = np.concatenate([
            class_indices[
                c
            ][
                start:end
            ]
            for c in classes
        ])


        rng.shuffle(
            test_idx
        )


        train_mask = np.ones(
            len(y),
            dtype=bool
        )

        train_mask[
            test_idx
        ] = False


        train_idx = all_idx[
            train_mask
        ]


        assert len(
            np.intersect1d(
                train_idx,
                test_idx
            )
        ) == 0


        splits.append(
            (
                train_idx,
                test_idx
            )
        )


    return splits


def print_splits(
    splits,
    groups,
    name
):
    print(
        "\n"
        + name
    )

    for i, (
        tr,
        te
    ) in enumerate(
        splits,
        1
    ):

        print(
            f"Fold {i}: "
            f"train specimens "
            f"{sorted(np.unique(groups[tr]).tolist())} "
            f"-> test specimens "
            f"{sorted(np.unique(groups[te]).tolist())}"
        )