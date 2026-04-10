<img width="2736" height="1706" alt="image" src="images/show.png" />

# MSM Agent Pipeline
An interactive human-in-the-loop pipeline for Markov state model construction.
The agent have hard coded sequential stages to run traditional pipeline using MSMbuilder.


# Setup
Install MSMbuilder first following [MSMbuilder documentation](https://github.com/msmbuilder/msmbuilder2022)
Clone the repo and go to the folder. Install and create environment, and set LLM API key locally.
```
pip install .
export OPENAI_API_KEY=your api key
```
```
python app_gradio_interact.py
```
    
