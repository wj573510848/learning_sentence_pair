from tensorflow.python.platform import gfile
import tensorflow as tf
import glob
import os
import re
import shutil
from tensorflow.contrib import predictor

pb_folder = './out_1_pb'

sub_folders = glob.glob(os.path.join(pb_folder,'*'))
sub_folder = ''
for i in sub_folders:
    if not re.search("temp", i):
        sub_folder = i
        break
assert sub_folder!=''
print(sub_folder)
pb_file = glob.glob(os.path.join(sub_folder,'*.pb'))[0]

predict_fn = predictor.from_saved_model(sub_folder)
graph_def = predict_fn.graph.as_graph_def()

log_dir = './log'
if os.path.isdir(log_dir):
    shutil.rmtree(log_dir)
os.makedirs(log_dir)

summaryWriter = tf.summary.FileWriter(log_dir,predict_fn.graph)

# tensorboard --logdir=log/ --port=6006
# 打开浏览器，可以查看网络结构