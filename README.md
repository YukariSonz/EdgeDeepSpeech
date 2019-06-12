# NCSDeepSpeech
This repo is for development for DeepSpeech on Neural Compute Stick


NOTE: This project is still on heavy development

The Convertion from .gb tensorflow file to .xml+.bin Intermediate Representation can be found here
https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_DeepSpeech_From_Tensorflow.html

Note: When converting the model, remember to selecte FP16 for the data type as NCS doesn't support FP32 (the default value)

###Current Issue
Memory: Due to limited memory on Neural Compute Stick and Complex speech to text model
