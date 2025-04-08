# import medigan and initialize Generators
from medigan import Generators
generators = Generators()

# generate 8 samples with model 8 (00008_C-DCGAN_MMG_MASSES). 
# Also, auto-install required model dependencies.
generators.generate(model_id=21, num_samples=8, install_dependencies=True)

