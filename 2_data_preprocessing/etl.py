import pickle
from pathlib import Path
from typing import Tuple, List, Optional
import logging
import pandas as pd

import great_expectations as gx
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import TargetEncoder

from thefuzz import fuzz, process

from great_expectations.data_context import FileDataContext
from great_expectations.datasource.fluent import BatchRequest
from great_expectations.checkpoint.types.checkpoint_result import CheckpointResult


# Random seed for making the assignments reproducible
RANDOM_SEED = 42

### COPY THE NEEDED FUNCTIONS HERE

def etl(path: Path, 
        gx_context_root_dir: Path, 
        gx_datasource_name: str, 
        gx_checkpoint_name: str,
        gx_expectation_suite_name: str,
        gx_run_name: str,
        feature_file_name: str,
        encoder_file_name: str, 
        target_file_name: Optional[str]=None, 
        fit_encoder: bool=False, 
        targets_included: bool=True
    ):
    ### START CODE HERE
    raise NotImplementedError()
    ### END CODE HERE

