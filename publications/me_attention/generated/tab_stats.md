| Track | Dataset | Ours | vs. | n | \bar d [95% CI] | W/T/L | p_{W} | p_{NB} | p_{W}^{Holm} |
|---|---|---|---|---|---|---|---|---|---|
| spectral | Collagen Pepsin | SmartAttn v0 | LR | 15 | -9.9 [-14.3, -5.5] | 2/0/13 | 0.001 | 0.172 | 0.005 |
| spectral | Collagen Pepsin | ME-Attn (meas.) | LR | 15 | -5.2 [-10.5, +0.1] | 4/0/11 | 0.095 | 0.536 | 0.189 |
| spectral | Collagen Pepsin | ex tokens v2* | LR | 15 | +0.8 [-1.5, +3.1] | 9/0/6 | 0.454 | 0.823 | 0.454 |
| spectral | Collagen Pepsin | hybrid conv attn | LR | 15 | -10.6 [-16.2, -5.6] | 4/0/11 | 0.003 | 0.221 | 0.010 |
| spectral | Collagen Pepsin | ex tokens v2* | RBF-SVM | 15 | +3.8 [+1.2, +6.4] | 12/0/3 | 0.010 | 0.359 | 0.031 |
| spectral | Collagen Pepsin | ex tokens v2* | RF | 15 | +7.5 [+3.9, +11.0] | 13/0/2 | 0.003 | 0.193 | 0.016 |
| spectral | Collagen Pepsin | ex tokens v2* | HGB | 15 | +6.5 [+3.1, +9.6] | 12/0/3 | 0.004 | 0.211 | 0.021 |
| spectral | Collagen Pepsin | ex tokens v2* | kNN | 15 | +9.8 [+7.6, +12.6] | 15/0/0 | <0.001 | 0.024 | <0.001 |
| spectral | Collagen Pepsin | ex tokens v2* | MLP | 15 | +7.9 [+2.3, +14.1] | 11/0/4 | 0.026 | 0.398 | 0.051 |
| spectral | Collagen Pepsin | ex tokens v2* | 1D-CNN | 15 | +11.9 [+7.5, +16.6] | 14/0/1 | <0.001 | 0.113 | 0.002 |
| spectral | Collagen Pepsin | ex tokens v2* | SpectralFormer | 15 | +7.2 [+3.1, +11.6] | 11/0/4 | 0.005 | 0.293 | 0.021 |
| spectral | Lichens 1 | SmartAttn v0 | SpectralFormer | 4 | -0.7 [-1.3, -0.0] | 1/0/3 | 0.250 | 0.329 | 0.750 |
| spectral | Lichens 1 | ME-Attn (meas.) | SpectralFormer | 4 | -1.3 [-2.2, -0.4] | 1/0/3 | 0.250 | 0.237 | 0.750 |
| spectral | Lichens 1 | ex tokens v2* | SpectralFormer | 4 | -0.6 [-1.3, -0.1] | 1/0/3 | 0.250 | 0.306 | 0.750 |
| spectral | Lichens 1 | hybrid conv attn | SpectralFormer | 4 | -4.0 [-8.6, -1.1] | 0/0/4 | 0.125 | 0.326 | 0.500 |
| spectral | Lichens 1 | ex tokens v2* | LR | 4 | -0.3 [-1.0, +0.3] | 1/0/3 | 0.375 | 0.619 | 1.000 |
| spectral | Lichens 1 | ex tokens v2* | RBF-SVM | 4 | -0.5 [-0.8, -0.1] | 1/0/3 | 0.250 | 0.254 | 1.000 |
| spectral | Lichens 1 | ex tokens v2* | RF | 4 | +0.2 [-0.2, +0.7] | 3/0/1 | 0.625 | 0.633 | 1.000 |
| spectral | Lichens 1 | ex tokens v2* | HGB | 4 | +1.3 [-0.7, +4.8] | 1/0/3 | 0.875 | 0.646 | 1.000 |
| spectral | Lichens 1 | ex tokens v2* | kNN | 4 | +7.0 [+2.9, +12.7] | 4/0/0 | 0.125 | 0.216 | 1.000 |
| spectral | Lichens 1 | ex tokens v2* | MLP | 4 | -0.0 [-0.5, +0.5] | 2/0/2 | 1.000 | 0.992 | 1.000 |
| spectral | Lichens 1 | ex tokens v2* | 1D-CNN | 4 | +0.8 [+0.5, +1.1] | 4/0/0 | 0.125 | 0.064 | 1.000 |
| spectral | Lichens 2 | SmartAttn v0 | RBF-SVM | 10 | -3.2 [-5.6, -1.0] | 3/0/7 | 0.027 | 0.458 | 0.059 |
| spectral | Lichens 2 | ME-Attn (meas.) | RBF-SVM | 10 | -3.5 [-5.3, -1.7] | 2/0/8 | 0.014 | 0.313 | 0.055 |
| spectral | Lichens 2 | ex tokens v2* | RBF-SVM | 10 | -0.1 [-1.0, +0.8] | 6/0/4 | 1.000 | 0.950 | 1.000 |
| spectral | Lichens 2 | hybrid conv attn | RBF-SVM | 10 | -4.2 [-10.0, -0.7] | 2/0/8 | 0.020 | 0.657 | 0.059 |
| spectral | Lichens 2 | ex tokens v2* | LR | 10 | +3.1 [+1.3, +4.9] | 9/0/1 | 0.014 | 0.361 | 0.082 |
| spectral | Lichens 2 | ex tokens v2* | RF | 10 | +3.2 [+1.7, +4.6] | 9/0/1 | 0.006 | 0.238 | 0.041 |
| spectral | Lichens 2 | ex tokens v2* | HGB | 10 | +1.9 [+0.6, +3.1] | 8/0/2 | 0.020 | 0.432 | 0.098 |
| spectral | Lichens 2 | ex tokens v2* | kNN | 10 | +2.9 [+1.2, +4.7] | 7/0/3 | 0.027 | 0.374 | 0.109 |
| spectral | Lichens 2 | ex tokens v2* | MLP | 10 | +0.9 [-0.1, +2.0] | 6/0/4 | 0.275 | 0.656 | 0.826 |
| spectral | Lichens 2 | ex tokens v2* | 1D-CNN | 10 | +4.9 [+3.8, +5.9] | 10/0/0 | 0.002 | 0.029 | 0.016 |
| spectral | Lichens 2 | ex tokens v2* | SpectralFormer | 10 | +0.4 [-0.3, +1.2] | 6/0/4 | 0.375 | 0.776 | 0.826 |
| spatial | Collagen Pepsin | ME-Attn (spatial)* | Patch-mean RF | 6 | +1.4 [-6.1, +7.7] | 4/0/2 | 0.688 | 0.862 | 0.688 |
| spatial | Collagen Pepsin | ME-Attn (spatial)* | Patch-mean LR | 6 | +6.0 [-0.6, +13.1] | 3/0/3 | 0.438 | 0.483 | 1.000 |
| spatial | Collagen Pepsin | ME-Attn (spatial)* | 2D-CNN | 6 | +4.9 [-3.7, +14.5] | 3/0/3 | 0.688 | 0.656 | 1.000 |
| spatial | Collagen Pepsin | ME-Attn (spatial)* | SpectralFormer-P | 6 | +3.5 [-6.7, +14.5] | 5/0/1 | 0.312 | 0.767 | 1.000 |
| spatial | Collagen Pepsin | ME-Attn (spatial)* | SSFTT | 6 | +2.8 [-7.4, +13.7] | 2/0/4 | 1.000 | 0.821 | 1.000 |
| spatial | Lichens 1 | ME-Attn (spatial)* | 2D-CNN | 4 | -0.6 [-1.0, -0.1] | 0/0/4 | 0.125 | 0.254 | 0.125 |
| spatial | Lichens 1 | ME-Attn (spatial)* | Patch-mean LR | 4 | -0.3 [-0.8, +0.2] | 2/0/2 | 0.625 | 0.581 | 1.000 |
| spatial | Lichens 1 | ME-Attn (spatial)* | Patch-mean RF | 4 | -0.3 [-0.9, +0.4] | 2/0/2 | 0.625 | 0.695 | 1.000 |
| spatial | Lichens 1 | ME-Attn (spatial)* | SpectralFormer-P | 4 | +0.3 [-0.4, +1.0] | 3/0/1 | 0.625 | 0.667 | 1.000 |
| spatial | Lichens 1 | ME-Attn (spatial)* | SSFTT | 4 | -0.5 [-1.4, -0.0] | 1/0/3 | 0.250 | 0.504 | 1.000 |
| spatial | Lichens 2 | ME-Attn (spatial)* | 2D-CNN | 4 | -1.5 [-3.6, -0.1] | 0/0/4 | 0.125 | 0.577 | 0.125 |
| spatial | Lichens 2 | ME-Attn (spatial)* | Patch-mean LR | 4 | +1.6 [+0.3, +3.6] | 4/0/0 | 0.125 | 0.548 | 0.625 |
| spatial | Lichens 2 | ME-Attn (spatial)* | Patch-mean RF | 4 | +1.8 [-0.2, +4.3] | 3/0/1 | 0.250 | 0.593 | 0.625 |
| spatial | Lichens 2 | ME-Attn (spatial)* | SpectralFormer-P | 4 | +0.6 [-1.7, +2.5] | 3/0/1 | 0.875 | 0.845 | 0.875 |
| spatial | Lichens 2 | ME-Attn (spatial)* | SSFTT | 4 | -0.5 [-0.8, -0.1] | 0/0/4 | 0.125 | 0.396 | 0.625 |
