
# AlexNet Retraining - Australian Politicians Recognition
## Dependencies
Codes in this folder depends on the following libraries:
- tensorflow 1.13.1
- numpy 1.16.3
- opencv  4.1.0
- dlib 19.4 
- scikit-image 0.15.0
> Use `conda install` to install missing libraries
## Training
1. In [printpath.py](https://github.com/HanwenZheng/PoliticiansAU_Recognition/blob/master/Alexnet/printpath.py "printpath.py") , set `path` to your Dataset's location and run it.
2. In [finetune.py](https://github.com/HanwenZheng/PoliticiansAU_Recognition/blob/master/Alexnet/finetune.py "finetune.py") ,  set `train_file` `val_file` to the two newly generated text files. set `filewriter_path` `checkpoint_path` to desired log file and checkpoint file locations. [Optional, specify hyperparameters like `learning_rate` `num_epochs` `batch_size` `train_layers`.] Set `num_classes` equal to your Dataset's number of distinct labels. Run it.
3. [Optional] Follow the direction to open Tensorboard.
4. Training should take several minutes to hours depending on your Dataset's size to finish.
## Verify
1. Run [get_test_class.py](https://github.com/HanwenZheng/PoliticiansAU_Recognition/blob/master/Alexnet/get_test_class.py "get_test_class.py") and get `class_names` .
1. In [testrun.py](https://github.com/HanwenZheng/PoliticiansAU_Recognition/blob/master/Alexnet/testrun.py "testrun.py") , set `class_names` to the one you just got from last step. Set `image_dir` to your test image folder. Set `checkpoint_path` to your latest .ckpt file. 
2. Run it and observe result.
## Freeze Graph
python freeze_graph.py \
--input_graph=`path_to_graph` \
--input_checkpoint=`path_to_ckpt` \
--input_binary=false \
--output_graph=`path_to_save_graph` \
--output_node_names=softmax_tensor
## Optimize for Inference
python optimize_for_inference.py \
--input=`path_to_graph` \
--output=`path_to_save_graph` \
--frozen_graph=True \
--input_names=input_tensor \
--output_names=softmax_tensor
