# AI-Handwriting-Alzheimer-Diagnosis
Project for CS4100: Foundations of AI involving using an AI model to diagnose Alzheimer's Disease based on Handwriting Samples

<img width="932" height="705" alt="AI_Handwriting_Alzheimer_Diagnosis" src="https://github.com/user-attachments/assets/27283049-b5c2-4294-acee-aa70bbb0e54a" />

**Design Structure:**
- Since the Handwriting Sample Consists of two different types of data (Image Data and Tabular Data) then we can split it into two separate models
- Each section, Image Processing and Tabular Processing, are completely disjoint from each other and only come together in a multimodal fusion model. This way we can optionally have an image input or a CSV input, or both.
- There will be a CNN to process the incoming Image Sample, and Grad-CAM will be looped in during the convolution stage to create a heat map of the image to explain the model's decisions
- There will be a MLP to process the Tabular Sample from the CSV
- Both are fed into a fusion model which will use there output to produce a combined output and more accurate diagnosis
- Results are processed and displayed in a GUI. In the case an Image Sample was used, then the Grad-CAM output will also be displayed in the GUI

**File Structure:**
- All components of training, executing, and testing each model should be in their own respective files (i.e. training.py, execute.py, testing.py, ...).

**Important Notes:**
- The models should be trained **per task** meaning that each time a sample is input to a model, an indicator for which task it is associated with should be included (consists of a `taskId`). From a user perspective, the GUI can have a drop for Task 1-25 to select the task the sample is associated with in the DARWIN dataset.
- Completing the CNN and Grad-CAM portion (Image Processing Section) is the top priority and marks the MVP of this project. This should be completed as soon as possible.

**Demo Data:**
- A selection of sample data is provided in the /demo directory for testing purposes. The available data includes: 
    - /images: original images for use in the CNN and Fusion model
    - /csv/patient: patient kinematic data for use in the standalone MLP
    - /csv/task: patient/task-specific kinematic data for use in the Fusion model
- All files are labeled using patient global IDs. The ID ranges correspond to the following groups:
    - 1-89: Alzheimer's disease
    - 90-174: Healthy control