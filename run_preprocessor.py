#!/usr/bin/env python3
"""Launch the MEHSI Preprocessor application."""

from mehsi_preprocessor.app import main

if __name__ == "__main__":
    main()

    # python
    # experiments / run_master_experiment.py \
    # - -data - dir
    # "Data/processed/Sponges Acid Group 1" \
    # - -output
    # "results/Sponges_Acid_Group1" \
    # - -retrain \
    # - -n - bands
    # 5, 10, 20, 30, 50 \
    #                - -n - dims
    # 1, 3