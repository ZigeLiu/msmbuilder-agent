# create a new conda environment
conda create -n agent python=3.11
conda activate agent

# install msmbuilder agent
pip install ./msmbuilder-agent

# install msmbuilder2022
git clone https://github.com/msmbuilder/msmbuilder2022.git
python -m pip install ./msmbuilder2022

# set up api key
conda env config vars set OPENAI_API_KEY=your api key
#conda env config vars set GOOGLE_API_KEY=your api key
conda activate agent