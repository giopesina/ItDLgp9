# import medigan and initialize Generators
from medigan import Generators
generators = Generators()

# generate 8 samples with model 8 (00008_C-DCGAN_MMG_MASSES). 
# Also, auto-install required model dependencies.
generators.generate(model_id=2, num_samples=32, install_dependencies=True)
generators.generate(model_id=5, num_samples=32, install_dependencies=True)

