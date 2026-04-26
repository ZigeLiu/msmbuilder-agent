<img width="2736" height="1706" alt="image" src="images/show.png" />

# MSM Agent Pipeline
An interactive human-in-the-loop pipeline for Markov state model construction.  
The agent have hard coded sequential stages to run traditional pipeline using MSMbuilder.
```
Stage 1: MD simulations → features
        ↓
Stage 2: scan parameter for tICA/dimensionality reduction (lagtime, number of components, and etc.)
        ↓
Stage 3: set optimal parameter and fit tICA, features → tICs
        ↓
Stage 4: geometric clustering, tICs → cluster labels 
        ↓
Stage 5: scan parameter for Markov State Model construction (lagtime, number of components, and etc.)
        ↓
Stage 6: set optimal parameter and construct Markov State Model, cluster labels → microstate MSM
        ↓
Stage 7: kinetic lumping into a few states and model evaluation, microstate MSM → macrostate MSM
```


# Setup
If working with conda, directly running setup.sh will create new environment, install all required packages and setup API key. Please add your api key before running.
```
bash setup.sh
```
If using other platforms, manual installation as following. Install MSMbuilder first following [MSMbuilder documentation](https://github.com/msmbuilder/msmbuilder2022).
Clone the agent repo and go to the folder. Create new virtual environment and install package, and set LLM API key for openai/google in your terminal.
```
pip install .
export OPENAI_API_KEY=your api key
export GOOGLE_API_KEY=your api key
```
If runing with local LLM with ollama, please make sure your ollama server have been launched in another terminal. Change model name in agent_ollama.py to the model you are using.  
Running the agent.
```
python agent_openai.py
```
Follow link to open user interface in browser.
