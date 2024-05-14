# InsectSound1000Tools
A repository of different scripts for sound data processing that were used to create the InsectSound1000 dataset, available at: https://doi.org/10.5073/20231024-173119-0, and described in the accompanying paper.

To run the "SampleExtractor_InsectSound1000" script, open up the script and change the relevant parameters at the very bottom of the file. Set your input_dic and target_dic path, specify the recording dates and adjust all other variables to your liking. The script reads TDMS recording files from folders named with their recording date found in input_dic. The script performs some basic activity detection on a prefiltered signal. The output of the script is equal length, non-overlapping sound samples down sampled to 16 kHz, in WAVE file format at 32-bit resolution.

"Split_DataSet_byDates_universal" is a Jupyter Notebook designed to split the InsectSound1000 dataset.  

"fetchfiles" contains a helper function used by booth programs to get a list of all files in a given directory that contain a given string in their filename.

Please cite:

Branding, J., von Hörsten, D., Böckmann, E. et al. 
InsectSound1000 An insect sound dataset for deep learning based acoustic insect recognition. 
Sci Data 11, 475 (2024). 
https://doi.org/10.1038/s41597-024-03301-4
