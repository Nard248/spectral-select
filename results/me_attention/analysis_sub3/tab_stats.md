| Track | Dataset | Ours | vs. | n | \bar d [95% CI] | W/T/L | p_{W} | p_{NB} | p_{W}^{Holm} |
|---|---|---|---|---|---|---|---|---|---|
| spectral | Lichens 2 (3-class subset) | SmartAttn v0 | RBF-SVM | 25 | -1.8 [-2.3, -1.2] | 1/0/24 | <0.001 | 0.027 | <0.001 |
| spectral | Lichens 2 (3-class subset) | ME-Attn (meas.) | RBF-SVM | 25 | -2.1 [-3.1, -1.2] | 5/0/20 | <0.001 | 0.137 | <0.001 |
| spectral | Lichens 2 (3-class subset) | ex tokens v2 | RBF-SVM | 25 | -0.9 [-2.0, -0.0] | 7/0/18 | 0.113 | 0.520 | 0.227 |
| spectral | Lichens 2 (3-class subset) | ex tokens + linear* | RBF-SVM | 25 | -0.6 [-1.4, -0.0] | 8/0/17 | 0.141 | 0.510 | 0.227 |
| spectral | Lichens 2 (3-class subset) | hybrid conv attn | RBF-SVM | 25 | -2.1 [-2.7, -1.5] | 1/0/24 | <0.001 | 0.025 | <0.001 |
| spectral | Lichens 2 (3-class subset) | ex tokens + linear* | LR | 25 | +2.5 [+0.6, +4.6] | 16/0/9 | 0.039 | 0.393 | 0.197 |
| spectral | Lichens 2 (3-class subset) | ex tokens + linear* | RF | 25 | +1.1 [+0.4, +1.8] | 21/0/4 | 0.002 | 0.256 | 0.013 |
| spectral | Lichens 2 (3-class subset) | ex tokens + linear* | HGB | 25 | +0.1 [-0.7, +0.9] | 16/0/9 | 0.508 | 0.916 | 1.000 |
| spectral | Lichens 2 (3-class subset) | ex tokens + linear* | kNN | 25 | +0.7 [+0.1, +1.2] | 19/0/6 | 0.007 | 0.384 | 0.040 |
| spectral | Lichens 2 (3-class subset) | ex tokens + linear* | MLP | 25 | +0.3 [-0.5, +1.0] | 16/0/9 | 0.200 | 0.760 | 0.601 |
| spectral | Lichens 2 (3-class subset) | ex tokens + linear* | 1D-CNN | 25 | +2.5 [+1.7, +3.4] | 22/0/3 | <0.001 | 0.044 | <0.001 |
| spectral | Lichens 2 (3-class subset) | ex tokens + linear* | SpectralFormer | 25 | +0.0 [-1.1, +1.1] | 11/0/14 | 0.731 | 0.991 | 1.000 |
