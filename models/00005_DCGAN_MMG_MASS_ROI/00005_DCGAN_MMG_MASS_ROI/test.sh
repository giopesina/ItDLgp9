#! /bin/bash

echo "Running a test: Generate method of this medigan model module."

echo "1. Creating and activating virtual environment called MMG_env."
python3 -m venv MMG_env
source MMG_env/bin/activate

echo "2. Pip install dependencies from requirements.txt"
pip install -r requirements.txt

echo "3. Run the generate function with parameters"

python -c "from __init__ import generate; 
model_file='500.pt';
num_samples=10;
output_path='images/';
save_images=True;
generate(model_file=model_file, num_samples=num_samples, output_path=output_path, save_images=save_images)"

echo "4. Any errors? Have synthetic images been successfully created in folder /images?"