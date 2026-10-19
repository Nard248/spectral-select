# Sweep 'night' (launched Mon Oct 19 00:43:44 BST 2026)
Replaces 'final' (too slow under CPU contention from another project's RL run, load ~560: 4 jobs in 16 min).
Same models, preprocessing and training loop as FROZEN_SPACES.md; reduced: budget T=3 (default + 2 random,
successive halving), max epochs 25 (patience per harness default), repeats pepsin=2 (6 units), lichens2=2 (4 units),
lichens1=1 (4 units). Env: PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.25, PYTORCH_MPS_LOW_WATERMARK_RATIO=0.2.
Command: see COMMAND. PID: see PID.
Extension (decided at 02:45 after seeing the first 2 repeats of the spectral track, to gain statistical power; applied
equally to all 12 spectral models): repeats 2 -> 5 for collagen_pepsin (15 units) and lichens2 (10 units), via extend.sh,
which starts automatically when the main run ends. Lichens 1 not extended (accuracies ~99%, near ceiling; most expensive).
extension start Mon Oct 19 04:49:53 BST 2026
lichens1/spatial_attn 4 jobs failed (worker OOM-killed at end of main run, workers had grown to 31-49 GB); queued for rerun after extension with max_tasks_per_child=3.
extension end Mon Oct 19 05:20:17 BST 2026
extension relaunched Mon Oct 19 05:20:19 BST 2026 after deadlock (max_tasks_per_child in py3.11.0); option removed
rerun lichens1/spatial_attn start Mon Oct 19 06:43:43 BST 2026
rerun lichens1/spatial_attn end Mon Oct 19 06:46:54 BST 2026
