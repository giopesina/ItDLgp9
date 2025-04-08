#! /bin/bash

echo "Running a test: Generate method of this medigan model module."

echo "1. Creating and activating virtual environment called MMG_env."
python3 -m venv MMG_env
source MMG_env/bin/activate

echo "2. Pip install dependencies from requirements.txt"
pip3 install --upgrade pip --quiet
pip3 install -r requirements.txt
#pip3 install -f https://download.pytorch.org/whl/torch_stable.html torch==1.8.1+cu111 torchvision==0.9.1+cu111 

echo "3. Run the generate function with parameters"

python3 -c "from __init__ import generate; 
model_file='netG_T1toT2_checkpoint.pth.tar';
input_path='T1';
output_path='examples';
save_images=True;
num_samples=10;
translate_all_images=False;
gpu_id = 0;
T1_to_T2=False;

generate(
	model_file=model_file, 
	input_path=input_path, 
	output_path=output_path,
	save_images=save_images,
	num_samples=num_samples,
	gpu_id=gpu_id, 
	translate_all_images=translate_all_images,
	T1_to_T2=T1_to_T2)"

echo "4. Any errors? Have synthetic images been successfully created in folder /output_folder?"


