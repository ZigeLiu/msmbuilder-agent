<img width="2736" height="1706" alt="image" src="images/show1.png" />

# MSM Agent Pipeline
An interactive human-in-the-loop pipeline for Markov state model construction.  
The agent has hard-coded sequential stages to run traditional pipelines using MSMbuilder. LLM's reasoning capability is used for understanding the topology and suggest on features to use, inspecting model quality test results and providing parameter tuning suggestions.
```
Stage 1: MD simulations → features
        ↓
Stage 2: scan parameters for tICA/dimensionality reduction (lagtime, number of components, etc.)
        ↓
Stage 3: set optimal parameters and fit tICA, features → tICs
        ↓
Stage 4: geometric clustering, tICs → cluster labels
        ↓
Stage 5: scan parameters for Markov State Model construction (lagtime, number of components, etc.)
        ↓
Stage 6: set optimal parameters and construct Markov State Model, cluster labels → microstate MSM
        ↓
Stage 7: kinetic lumping into a few states and model evaluation, microstate MSM → macrostate MSM
```


# Setup
If working with conda, directly running setup.sh after cloning the repo to your local folder. This will create new environment, install all required packages and setup API key. Please add your api key before running.
```
bash setup.sh
```
If using other platforms, manual installation is as follows: 
- Install MSMbuilder first by following [MSMbuilder documentation](https://github.com/msmbuilder/msmbuilder2022).
- Clone the agent repo and go to the folder. 
- Create virtual environment.
- Follow the following command to install MSMBuilding agent and set LLM API key for openai.
```
pip install .
export OPENAI_API_KEY=your_api_key_here
```
If running a local LLM via Ollama, please make sure the Ollama server is already running. Change the model name in agent_ollama.py to the model you are using ( currently removed, will add later). 

# Running human in the loop agent 
Running the agent with OpenAI or the model of your choice.
```
python agent_openai.py
```

# Running automatic parameter search agent
Running the agent with OpenAI.
```
python agent_auto.py
```
Follow link to open user interface in browser. You can update settings in config editor or tell the agent to update for you. All run output are saved to `/results/` inside the install folder.