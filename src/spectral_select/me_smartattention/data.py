import json
import pickle
import gc
import numpy as np

from PIL import Image
from scipy.ndimage import label


def load_dataset(
    data_dir,
    min_specimen_pixels=100
):
    """
    Load and prepare the ME-HSI dataset.

    Small disconnected class-mask components are discarded
    before any labels, groups, or spectra are created.

    Parameters
    ----------
    data_dir : Path
        Dataset directory.

    min_specimen_pixels : int
        Minimum connected-component size required to count
        as a real specimen.

    Returns
    -------
    dict
        Prepared dataset.
    """

    # ========================================================
    # CLASS MASK
    # ========================================================

    with open(
        data_dir / "roi_regions.json",
        "r"
    ) as f:
        roi = json.load(f)

    rgb = np.array(
        Image.open(
            data_dir / "class_mask.png"
        ).convert("RGB")
    )

    class_map = np.full(
        rgb.shape[:2],
        -1,
        dtype=np.int16
    )

    for cls, info in enumerate(
        roi["classes"]
    ):
        color = np.asarray(
            info["color"]
        )

        class_map[
            np.all(
                rgb == color,
                axis=2
            )
        ] = cls


    # ========================================================
    # REAL SPECIMENS
    # ========================================================

    specimen_map = np.full(
        class_map.shape,
        -1,
        dtype=np.int16
    )

    specimen_classes = []

    sid = 0
    ignored_pixels = 0

    for cls in range(
        len(roi["classes"])
    ):

        components, n = label(
            class_map == cls,
            structure=np.ones(
                (3, 3),
                dtype=np.uint8
            )
        )

        class_specimens = 0

        for component in range(
            1,
            n + 1
        ):

            m = (
                components
                == component
            )

            size = int(
                m.sum()
            )

            # -----------------------------------------------
            # CRITICAL:
            # reject tiny disconnected mask artifacts
            # -----------------------------------------------

            if size < min_specimen_pixels:
                ignored_pixels += size
                continue

            specimen_map[m] = sid
            specimen_classes.append(cls)

            sid += 1
            class_specimens += 1

        print(
            f"Class {cls}: "
            f"{class_specimens} specimens"
        )


    # ========================================================
    # FINAL FOREGROUND
    #
    # Only pixels belonging to accepted specimens.
    # ========================================================

    foreground = (
        specimen_map >= 0
    )

    y = class_map[
        foreground
    ].astype(
        np.int64
    )

    groups = specimen_map[
        foreground
    ].astype(
        np.int64
    )

    specimen_classes = np.asarray(
        specimen_classes,
        dtype=np.int64
    )


    # Safety checks
    assert np.all(
        groups >= 0
    )

    assert len(
        np.unique(groups)
    ) == sid

    assert all(
        np.unique(
            y[groups == s]
        ).size == 1
        for s in np.unique(groups)
    )


    # ========================================================
    # SPECTRAL DATA
    # ========================================================

    with open(
        data_dir / "spectra_masked.pkl",
        "rb"
    ) as f:
        spectral_data = pickle.load(f)

    # Ensure all selected specimen pixels have spectral data
    if "mask" in spectral_data:

        spectral_mask = np.asarray(
            spectral_data["mask"]
        ).astype(bool)

        assert np.all(
            spectral_mask[
                foreground
            ]
        ), (
            "Some accepted specimen pixels "
            "do not have spectral measurements."
        )


    excitations = np.asarray(
        spectral_data[
            "excitation_wavelengths"
        ],
        dtype=float
    )

    entries = []


    for excitation in excitations:

        entry = spectral_data[
            "data"
        ][
            str(float(excitation))
        ]

        cube = entry[
            "cube"
        ]

        emission = None

        for key, value in entry.items():

            arr = np.asarray(
                value
            )

            if (
                "emission"
                in key.lower()
                and arr.ndim == 1
            ):
                emission = arr.astype(
                    float
                )
                break

        entries.append(
            (
                cube,
                emission
            )
        )


    # ========================================================
    # GLOBAL EMISSION AXIS
    # ========================================================

    if all(
        emission is not None
        for _, emission in entries
    ):

        emissions = np.array(
            sorted({
                round(
                    float(w),
                    6
                )
                for _, emission in entries
                for w in emission
            }),
            dtype=float
        )

        emission_lookup = {
            round(
                float(w),
                6
            ): i
            for i, w in enumerate(
                emissions
            )
        }

    else:

        emissions = np.arange(
            max(
                cube.shape[-1]
                for cube, _ in entries
            ),
            dtype=float
        )

        emission_lookup = None


    # ========================================================
    # EEM REPRESENTATION
    # ========================================================

    E = len(
        excitations
    )

    M = len(
        emissions
    )

    N = int(
        foreground.sum()
    )


    X_eem = np.zeros(
        (
            N,
            E,
            M
        ),
        dtype=np.float32
    )

    eem_mask = np.zeros(
        (
            E,
            M
        ),
        dtype=bool
    )


    for e, (
        cube,
        emission
    ) in enumerate(
        entries
    ):

        X = cube[
            foreground
        ].astype(
            np.float32,
            copy=False
        )

        if emission is not None:

            positions = np.array([
                emission_lookup[
                    round(
                        float(w),
                        6
                    )
                ]
                for w in emission
            ])

        else:

            positions = np.arange(
                M - X.shape[1],
                M
            )


        X_eem[
            :,
            e,
            positions
        ] = X

        eem_mask[
            e,
            positions
        ] = True


    # ========================================================
    # LR INPUT
    #
    # Remove nonexistent EEM positions completely.
    # ========================================================

    X_flat = X_eem[
        :,
        eem_mask
    ]


    # ========================================================
    # SUMMARY
    # ========================================================

    print(
        f"\nPixels: {len(y):,}"
    )

    print(
        "Specimens:",
        len(
            np.unique(groups)
        )
    )

    print(
        "Specimens/class:",
        np.bincount(
            specimen_classes
        ).tolist()
    )

    print(
        "Ignored artifact pixels:",
        ignored_pixels
    )

    print(
        "Classes:",
        len(
            np.unique(y)
        )
    )

    print(
        "Excitations:",
        E
    )

    print(
        "Emission positions:",
        M
    )

    print(
        "Valid measurements:",
        int(
            eem_mask.sum()
        )
    )

    print(
        "LR input:",
        X_flat.shape
    )

    print(
        "EEM input:",
        X_eem.shape
    )


    del spectral_data
    gc.collect()


    return {
        "X_flat":
            X_flat,

        "X_eem":
            X_eem,

        "eem_mask":
            eem_mask,

        "y":
            y,

        "groups":
            groups,

        "foreground":
            foreground,

        "specimen_map":
            specimen_map,

        "specimen_classes":
            specimen_classes,

        "class_map":
            class_map,

        "rgb":
            rgb,

        "excitations":
            excitations,

        "emissions":
            emissions,

        "ignored_pixels":
            ignored_pixels,
    }