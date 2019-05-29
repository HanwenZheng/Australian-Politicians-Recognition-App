# MobileNet Retraining - Australian Politicians Recognition
## Working Environment
Codes in this folder assumes you are using the following environment:

- Python 3.7.3
- Anaconda 3
## Dependencies
Codes in this folder depends on the following libraries:

- tensorflow 1.13.1
- numpy 1.16.3

> Use `conda install` to install missing libraries
## Training

python [retrain.py](https://github.com/HanwenZheng/PoliticiansAU_Recognition/blob/master/Mobilenet/retrain.py "retrain.py") \  --bottleneck_dir=`path_to_bottleneck` \  
--model_dir=`path_to_save_model` \  
--summaries_dir=`path_to_save_logfile` \  
--output_graph=`path_to_save_graph` \  
--output_labels=`path_to_save_labels` \  
--image_dir=`path_to_Dataset` \  
--how_many_training_steps=`training_steps` \  
--architecture=`pre_trained_model` (For example, "mobilenet_1.0_224")
## Convert to .tflite file - Float point 
tflite_convert \  --graph_def_file=`path_to_graph` \  
--output_file=`path_to_save_graph` \  
--input_format=TENSORFLOW_GRAPHDEF \  
--output_format=TFLITE \  
--input_shape=1,224,224,3 \  
--input_array=input \  
--output_array=final_result \  
--inference_type=FLOAT \  
--default_ranges_max=`num_class` \  
--input_data_type=FLOAT
## Convert to .tflite file - Quantized
tflite_convert \  --graph_def_file=`path_to_graph` \  
--output_file=`path_to_save_graph` \  
--input_format=TENSORFLOW_GRAPHDEF \  
--output_format=TFLITE \  
--input_shape=1,224,224,3 \  
--input_array=input \  
--output_array=final_result \  
--inference_type=QUANTIZED_UINT8 \  
--mean_values=224 \  
--std_dev_values=223 \  
--default_ranges_min=0 \  
--default_ranges_max=`num_class` \  
--input_data_type=FLOAT
