import os
from flyai.train_helper import submit, upload_data

# upload_data("./data/cats.zip", dir_name="/data", overwrite=True)
submit(train_name="train_cat_data", code_path=os.curdir, cmd="python train.py")
