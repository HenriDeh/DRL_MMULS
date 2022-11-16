This branch of this package accompagnies the following publication : . 

To install this package, run the following commands in the Julia REPL.
```
using Pkg
Pkg.add("url#single-item")
cd("path/of/installed/project")
Pkg.instantiate()
```
The experiments described in the article were performed using Julia 1.8.1. Any newer version should yield almost identical results. To use another julia version, remove the 1.8.1 compat entry in the Project.toml file.

To reproduce the results, run the julia files at `scripts/single-item/experiments/PPO_solve.jl` (train 320 agents with non-evolving demand), `scripts/single-item/experiments/PPO_solve_adi.jl` (train 320 agents with the MMFE), and `scripts/single-item/experiments/DP_solve.jl` (solve all problems with the baselines). Using the `include("path/of/the/file.jl")` command.
