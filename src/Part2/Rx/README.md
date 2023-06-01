This file contains minimal instructions to get the accompanying demos running with Jupyter Notebook, Julia and RxInfer.

# Install Julia
In order to install the Julia language (v1.8 or higher), follow the platform-specific instructions at https://julialang.org/downloads/platform.html

# Install Jupyter Notebook
Jupyter notebook is a framework for running Julia (among other languages) in a browser environment. It is especially well suited for showing demo applications and interactive experimentation (i.e. playing around). In order to install Jupyter Notebook, follow the instructions at https://jupyter.readthedocs.io/en/latest/install.html

# Install required packages
The demos require some packages to be imported in Julia. Open Julia
```
$ julia
```
and enter the package prompt by typing a closing bracket
```
julia> ]
```
Next, activate the virtual environment
```
(v1.8) pkg> activate .
```
and instantiate the required packages
```
(Rx) pkg> instantiate
```

# Run the demo
Exit Julia, navigate to the root directory and start a Jupyter server
```
~/Rx$ jupyter notebook
```
A browser window will open, and you can select the demo you wish to run.

# License
(c) 2023 BIASlab