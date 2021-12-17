This branch of this package accompagnies the following publication : . 

To install this package, run the following commands in the Julia REPL.
```
using Pkg
Pkg.add("url#single-item")
cd("path/of/installed/project")
Pkg.instantiate()
```
The experiments described in the article were performed using Julia 1.6.2. Any newer version should yield almost identical results.

To reproduce the results, run the julia file at `scripts/single-item/single-item.jl` using the `include("scripts/single-item/single-item.jl")` command.
