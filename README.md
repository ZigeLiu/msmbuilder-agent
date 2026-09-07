# MSMBuilding Agent Pipeline

![MSMBuilding Agent pipeline overview](images/fig1.png)

MSMBuilding Agent is an interactive, human-in-the-loop system for constructing Markov state models (MSMs) from molecular dynamics simulations. It combines a fixed, reproducible MSMBuilder workflow with an LLM that can:

- inspect molecular topology and suggest suitable features;
- interpret model-quality diagnostics and warnings;
- recommend parameter adjustments based on expert knowledge; and
- guide the pipeline from featurization through macrostate construction.

The numerical work remains organized as a seven-stage pipeline:

```text
Stage 1: MD trajectories -> molecular features
    |
Stage 2: Scan tICA parameters, including lag time and component count
    |
Stage 3: Select parameters and fit tICA (features -> tICs)
    |
Stage 4: Cluster the projected data (tICs -> cluster assignments)
    |
Stage 5: Scan MSM parameters, including lag time and timescale count
    |
Stage 6: Fit the microstate MSM (cluster assignments -> microstate model)
    |
Stage 7: Lump microstates and evaluate the model (microstates -> macrostates)
```

## Requirements

- Python 3.10 or later
- [MSMBuilder 2022](https://github.com/msmbuilder/msmbuilder2022)
- An OpenAI API key when using the OpenAI-backed agents or a running local ollama server
- Conda (recommended) or another Python environment manager 

## Installation
Clone the current repo to your local disk, add your data folder to `data`.

### Conda setup

Review `setup.sh` and replace the API-key placeholder before running it. The script creates a Python 3.11 Conda environment, installs this agent and MSMBuilder 2022, and configures the API key.

```bash
bash setup.sh
```

After installation, reactivate the environment so the configured environment variables are available:

```bash
conda activate agent
```

### Manual setup

Create and activate a virtual environment, then run the following commands from the repository root:

```bash
python -m pip install .
git clone https://github.com/msmbuilder/msmbuilder2022.git
python -m pip install ./msmbuilder2022
export OPENAI_API_KEY="your_api_key_here"
```

For persistent credentials, configure `OPENAI_API_KEY` through your shell or environment manager.

### Local LLM setup
First make sure your machine have ollama installed and running. If not, run the following script first to install ollama.
```bash
bash setup_ollama.sh
ollama serve
```
If executed succeffully, ollama server have been running on your machine. In a new terminal start a LLM model of your choice (here Qwen3.8).
```bash
export OLLAMA_MODEL="qwen3.8"
ollama run qwen3.8
```
The above can be setup on remote GPU cluster, and use the public link for interacting. 

## Run the human-in-the-loop agent

Start the interactive OpenAI agent with:

```bash
python agent_openai.py
```

If using ollama, run the following command in three different terminals.
```bash
ollama serve
ollama run qwen3.8
python agent_ollama.py
```

Open the Gradio URL printed in the terminal. In the interface, you can edit the YAML configuration directly or ask the agent to update supported parameters for you.

![Human-in-the-loop agent interface](images/fig2.png)

## Run the automatic search agent

Start the automatic pipeline and reviewer agents with:

```bash
python agent_auto.py
```

The pipeline agent executes and adjusts stages, while the reviewer agent inspects each result before the workflow proceeds.

![Automatic search agent interface](images/fig3.png)

## Results

Each run receives its own directory under `results/` by default. Depending on the completed stages, this directory contains the resolved configuration, intermediate artifacs, manifests, saved model, and generated figures.

The Gradio interfaces display the active configuration, current stage, latest summary, and available plots while the pipeline is running.

