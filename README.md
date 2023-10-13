# InsectSound1000Tools
A repository of different scripts for sound data processing used to create the InsectSound1000 dataset.

To run the SampleExtractor_InsectSound1000 script, open up the script and change the relevant parameters at the very bottom of the file. Set your input_dic and target_dic path, specify the recording dates and adjust all other variables to your liking.
The script reads tdms recording files from folders named with there recording date found in input_dic. The script performs some basic activity detection on a prefiltered signal. The output of the script are equal length, non-overlapping sound samples down sampled to 16 kHz, in wav-file format at 32 bit of resolution. 

Please cite:

Branding et al. (2023), Scientific Data, InsectSound1000 An Insect
Sound Dataset for Deep Learning based Acoustic Insect Recognition
