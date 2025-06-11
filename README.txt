-- Folders --
Intermadiate = contains .h5 file per trained model (model architecture and weights), as well as an .npz file with predicted SPM and test SPM
logs = contains output and error logs from jobs on the cluster (if cluster is used)
Results = contains a subfolder per model setup with model output, as well as overall summarizing information on model performance and best model

-- Files --
1. config.yml = to define input << user interaction for model configuration
2. main.py = to run the model
3. result_visualization.py = separate plot and map making script, not integrated into model pipeline

Other code are called for in main.py

.sh files are used for job submission to cluster

Duration of training for southern Rhine ROFI on Nvidia H100 GPU, 200 epochs = 28 minutes