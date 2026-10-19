# Random-pixel vs whole-specimen CV (collagen_acetic, dev set)

Balanced accuracy, 3 folds x 1 repeat, default configs, preprocessing area/global/log1p. Pixel = balanced random pixel split (make_pixel_splits, pixels of every specimen in train and test); specimen = whole-specimen class-covering folds. Train rows capped at 40000.

| model | random-pixel | whole-specimen | gap |
|---|---|---|---|
| lr | 97.9 +- 0.1 | 87.8 +- 3.9 | +10.1 |
| mlp | 98.4 +- 0.1 | 86.8 +- 3.8 | +11.6 |
| smartattention_v0 | 98.4 +- 0.1 | 88.3 +- 2.8 | +10.1 |
