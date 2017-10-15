import os
import glob
import argparse
from scripts import label_image
import shutil
import tensorflow as tf
import numpy as np


path = 'default_val'
modelfile = 'tf_files/retrained_graph.pb'
labelfile = 'tf_files/retrained_labels.txt'
input_layer = "Mul"
output_layer = "final_result"
input_height=299
input_width=299
input_mean=128
input_std=128,
parser = argparse.ArgumentParser()
parser.add_argument("--folder", help="relative path to image dir")
parser.add_argument("--tfolder", help="relative path to tf-folder")
args = parser.parse_args()

if args.folder:
	path = args.folder
if args.tfolder:
	modelfile = args.tfolder + '/retrained_graph.pb'
	labelfile = args.tfolder + '/retrained_labels.txt'
graph = label_image.load_graph(modelfile)
os.chdir(path)
dir_path = os.getcwd()
for file in glob.glob("*.png") + glob.glob("*.jpg") + glob.glob("*gif"):
	#results = label_image.runFunc(file_nam=file, model_fil=modelfile, label_fil=labelfile, grap=graph)
	input_name = "import/" + input_layer
	output_name = "import/" + output_layer
	input_operation = graph.get_operation_by_name(input_name);
	output_operation = graph.get_operation_by_name(output_name);
	t = label_image.read_tensor_from_image_file(file,
								input_height=input_height,
								input_width=input_width,
								input_mean=input_mean,
								input_std=input_std)


	with tf.Session(graph=graph) as sess:
		results = sess.run(output_operation.outputs[0],
							{input_operation.outputs[0]: t})
	results = np.squeeze(results)

	top_k = results.argsort()[-5:][::-1]
	labels = label_image.load_labels(labelfile)
	highestlikely = ''
	for i in top_k:
		print(labels[i], results[i])
		highestlikely = labels[top_k[-5]]
	print(highestlikely)
	if not os.path.exists(os.path.dirname(dir_path+'/'+'Sorted_'+highestlikely)):
		os.makedirs(os.path.join(dir_path, 'Sorted_'+highestlikely))
		shutil.move(dir_path+'/'+file, dir_path+'/'+ 'Sorted_'+highestlikely+'/'+file)
	else:
		#shutil.move(os.path.join(path, file), os.path.join(path, results+'/'+file))
		shutil.move(dir_path+'/'+file, dir_path+'/'+'Sorted_'+highestlikely+'/'+file)