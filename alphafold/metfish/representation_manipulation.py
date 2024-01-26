import pickle
import numpy as np


def modify_representations(prev: dict = None, method: str ='reinitialize'):
  """Modify the pair, position (structural), or MSA (first row only) representations 
  that will be used as inputs for the next recycling iteration of AlphaFold.

  Args:
      prev (dict): Dict of pair, position, and msa representations.
      method (str): Method to use to modify representations.

  Returns:
      dict: modified pair, position and msa representations
  """

  # load the raw representations if not provided
  prev = prev or load_prev_representations()

  # apply modification method (test examples, will eventually modify prev outputs based on SAXS data)
  if method == 'reinitialize':
     repr_modified = reinitialize(prev)
  elif method == 'add_noise':
     repr_modified =  add_noise(prev)
  
  return repr_modified


def load_prev_representations():
  """Load previous representations from intermediate files saved when running colabfold with the --save-all and --save-recycles flags"""
  filename = 'filename'  # TODO - grab actual file path
  with open(filename, 'rb') as f:
      data = pickle.load(f) 

  representations = {'prev_msa_first_row': data['representations']['single'],
                     'prev_pair': data['representations']['pair'],
                     'prev_pos': data['structure_module']['final_atom_positions']}

  return representations


def add_noise(prev):
    """Adds gaussian noise to the pair, structure, and MSA representations"""
    prev_with_noise = dict()
    rng = np.random.default_rng()

    for key, value in prev.items():
        noise = rng.standard_normal(value.shape).astype('float16')  # gaussian noise with μ = 0, σ = 1
        prev_with_noise[key] = value + noise

    return prev_with_noise


def reinitialize(prev):
    """Reinitializes the pair, structure, and MSA, representations to zero arrays"""

    L = np.shape(prev['prev_pair'])[0]
    
    zeros = lambda shape: np.zeros(shape, dtype=np.float16)
    prev = {'prev_msa_first_row': zeros([L,256]),
            'prev_pair':          zeros([L,L,128]),
            'prev_pos':           zeros([L,37,3])}
    
    return prev
