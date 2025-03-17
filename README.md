# Reactive Pluto Notebooks from article "Constructing structured tensor priors for Bayesian inverse problems" 

These reactive [Pluto.jl](https://github.com/fonsp/Pluto.jl) notebooks are the numerical experiments of the article "[Constructing structured tensor priors for Bayesian inverse problems](https://arxiv.org/abs/2406.17597)" by Kim Batselier.

There are basically two ways to run these notebooks:

## 1. Run .jl files on your own computer

The recommended way to run these notebooks is on your own computer. This, however, requires that you install Julia first. Here are instructions on how to get these notebooks running on your machine. Note that steps 1 up to 2 only need to be done once :). 

1. Download and install [Julia](https://julialang.org/).

2. In Julia press the "]" key to open the package manager and then run the command "add Pluto".

3. Once Pluto is installed press backspace to return to the prompt and run the command 
    
    Using Pluto

    Pluto.run()


4. Pluto will start in your browser, open the notebook you want to run and then click on "Run notebook code" in the upper right corner.

## 2. Run .jl files in the cloud via Binder

If installing Julia is not possible then these notebooks can be run in the cloud via [Binder](https://mybinder.org/) instead. Please note that this option might be quite slow so please be patient. Simply click on the hyperlinks below to run the notebooks. Once the notebook has opened, click on "Run notebook code" in the upper right corner.

[Application 1 - Sampling structured tensor priors](https://binder.plutojl.org/v0.19.36/open?url=https%253A%252F%252Fraw.githubusercontent.com%252FTUDelft-DeTAIL%252FAbTensors%252Fmain%252FAbTensors1.jl)


[Application 2 - Completion of a Hankel matrix from noisy measurements](https://binder.plutojl.org/v0.19.36/open?url=https%253A%252F%252Fraw.githubusercontent.com%252FTUDelft-DeTAIL%252FAbTensors%252Fmain%252FAbTensors2.jl)

[Application 3 - Bayesian learning of MNIST classifier](https://binder.plutojl.org/v0.19.36/open?url=https%253A%252F%252Fraw.githubusercontent.com%252FTUDelft-DeTAIL%252FAbTensors%252Fmain%252FAbTensors3.jl)