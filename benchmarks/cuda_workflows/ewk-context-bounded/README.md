# Bounded CUDA diagnostic metadata experiment

Baseline: exact #4503 runtime-base overlay from Della run16609714.
Candidate: only `_cuda_argument` replaced, retaining all algorithm/kernel changes.
High-level arrays/records use outer layout class and length rather than recursively
building/stringifying `.type`. Dictionaries continue to return `<dict>`.
Deferred CUDA diagnostic notes contain less type detail; values and execution
semantics are unchanged. This is an experimental overlay, not an upstream commit.

Local tests forbid high-level type/repr/str and RecordArray.form access for a
1000-field array; check nested arguments and no retained array ownership.
The original AST outside `_cuda_argument` is verified unchanged while patching.

Submit from /home/iosborne/Projects/ewkcoffea-della:
    sbatch --export=NONE ewk-context-bounded/run-della.sbatch

Runs unprofiled baseline/candidate CPU+CUDA full analysis (5x20k events), histogram
parity and warmed phase reports, then separately profiles/traces candidate
obj_basic_selection in chunk2 after real chunk1 warmup. Existing ewk-persistent
scripts are included. Profiles perturb execution; do not use as benchmark times.
The basic diagnostic wrapper also regenerates existing run16610845 phase reports.

Results: results/JOBID/context-bounded/{comparison.md,phases-candidate.md,
basic-diagnostics/host/cuda/basic-{focus.txt,statements.json,profile.txt},
basic-diagnostics/basic-nsys-summary.txt,basic-diagnostics/trace-parity.md}.
Full pstats callers help test whether to_vec cost follows error-context capture;
this causal attribution is a hypothesis until the paired experiment is run.
