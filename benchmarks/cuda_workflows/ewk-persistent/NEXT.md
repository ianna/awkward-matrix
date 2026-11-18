# Saved phase report and larger-chunk sweep

## Existing run (no GPU needed)

In the activated environment, from ewkcoffea-della:

```bash
set -o pipefail
python ewk-persistent/phases.py results/16605355/persistent \
  | tee results/16605355/persistent/phase-comparison.md
```

Share that Markdown report. It excludes the first chunk and reports top-level
phases separately from nested subphases, sorted by CUDA minus CPU median.
The underlying upstream phase boundaries are not all fenced, so these identify
candidates for deeper profiling rather than proving isolated GPU costs. No
phase times are added together. Original JSON traces and histogram results are
left intact.

## Obtain enough input

On LXPLUS with the working CMS proxy:

```bash
cd /tmp/yana
xrdcp -f \
  root://cmsxrootd.fnal.gov//store/user/kmohrman/vbs_vvh_gpu_repro/mega_rep_1m.root \
  mega_rep_1m.root
```

Only after xrdcp succeeds:

```bash
scp /tmp/yana/mega_rep_1m.root \
  iosborne@della-gpu.princeton.edu:/home/iosborne/Projects/ewkcoffea-della/data/
```

## On Della

```bash
cd /home/iosborne/Projects/ewkcoffea-della
# If this is a new login:
module load anaconda3/2026.7
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ewkcoffea_della_repro
python ewk-persistent/prepare_1m.py && \
  sbatch --export=NONE ewk-persistent/run-della-sweep.sbatch
```

The sweep uses one allocated A100, first 100k then 200k target chunk sizes.
Coffea may choose slightly different actual ranges; inspect summary.json.
Each backend processes the entire input (`chunks=0` omits the upstream maxchunks
option), verifies continuous coverage through the last entry and at least three
chunks in one PID, then compares the same CPU/CUDA ranges and histograms.

Results:

```bash
cat results/JOBID/persistent-sweep/100000/comparison.md
cat results/JOBID/persistent-sweep/100000/phase-comparison.md
cat results/JOBID/persistent-sweep/200000/comparison.md
cat results/JOBID/persistent-sweep/200000/phase-comparison.md
```

Both sizes use the existing metadata-only context overlay. First-use costs are
retained separately from later-chunk medians; later event throughput and actual
chunk lengths are recorded in summary.json. The runs include I/O and host
histogram filling; they are not device-only measurements. CPU precedes CUDA,
so host file-cache bias remains. The two sizes run sequentially; a failure or
parity mismatch stops the sweep. Worker RSS/pool snapshots and sampled GPU
memory are retained. Larger chunk GPU execution is not yet validated.
