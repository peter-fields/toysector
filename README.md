# toysector

This repository contains the code and materials for the paper “Understanding Energy-Based Modeling of Proteins via an Empirically Motivated Minimal Ground Truth Model” by Peter William Fields, Vudtiwat Ngampruetikorn, Rama Ranganathan, David J. Schwab, and Stephanie Palmer, presented at the Synergy of Scientific and Machine Learning Modeling Workshop, ICML 2023. OpenReview ID: vxn5QGPFyi, ( https://openreview.net/forum?id=vxn5QGPFyi )

---

For fitting energy-based models, this repo utilizies SpinModels.jl at ( https://github.com/wavengampruetikorn/SpinModels.jl )

---

This code base is using the Julia Language and [DrWatson](https://juliadynamics.github.io/DrWatson.jl/stable/)
to make a reproducible scientific project named
> toysector

To (locally) reproduce this project, do the following:

0. Download this code base. Notice that raw data are typically not included in the
   git-history and may need to be downloaded independently.
1. Open a Julia console and do:
   ```
   julia> using Pkg
   julia> Pkg.add("DrWatson") # install globally, for using `quickactivate`
   julia> Pkg.activate("path/to/this/project")
   julia> Pkg.instantiate()
   ```

This will install all necessary packages for you to be able to run the scripts and
everything should work out of the box, including correctly finding local paths.

The Jupyter notebooks use IJulia. This package is not in the project environment, so ensure that IJulia is installed in your global Julia environment if it is not already (e.g. via Pkg.add("IJulia")) before running the notebooks.
