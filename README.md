# Context

This repo is to deploy the openpose repository using controlnet in a docker environment

# Install 

activate the virtual environment

`source env/bin/activate`

deactivate the virtual environment

`deactivate`

```bash

python3 -m venv env

source env/bin/activate

deactivate

pip freeze > requirements.txt

pip install -r requirements.txt

```

Remember to use anaconda powershell for development on Windowns machine

run the following programs

python predict.py

python evoke_api.py
