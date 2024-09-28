import pandas as pd
import numpy as np


def fix_data_frame(data) -> pd.DataFrame:
  target = [key for key, item in data.items() if item.dtype == 'object']
  for col in target:
    data[col] = data[col].astype('category')
    data[col] = data[col].cat.codes
  return data