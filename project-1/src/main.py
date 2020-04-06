
# Part 2: Probabilistic Generative Model
from generative_model import run_generative_analysis
run_generative_analysis(seed=0, method='classification', base_name='A', lim=[-3.5, 3.5, -3.5, 4.5])
run_generative_analysis(seed=0, method='blobs', base_name='B', lim=[-4, 5, -1.5, 6.5])

# Part 3: Discriminative model
from discriminative_model import run_discriminative_analisys
run_discriminative_analisys()