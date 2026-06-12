# AI-Handwriting-Alzheimer-Diagnosis
Project for CS4100: Foundations of AI involving using an AI model to diagnose Alzheimer's Disease based on Handwriting Samples

<img width="932" height="705" alt="AI_Handwriting_Alzheimer_Diagnosis" src="https://github.com/user-attachments/assets/27283049-b5c2-4294-acee-aa70bbb0e54a" />

## Running the Frontend

The frontend is a [Flask](https://flask.palletsprojects.com/) web app (`src/frontend/app.py`) that lets you upload an image and/or CSV handwriting sample and returns a diagnosis.

From the repo root:

```bash
# Install dependencies
pip install -r requirements.txt

# Start the app
python src/frontend/app.py
```

Then open **http://127.0.0.1:5000** in your browser. The app runs in debug mode by default.

## Scripts

The `scripts/` directory contains helper shell scripts for running, training, and testing the models. Activate your virtual environment first, then run a script from the repo root, e.g. `./scripts/train_cnn.sh`. Any extra arguments are passed through to the underlying Python program.

| Script | Description |
| --- | --- |
| `run_app.sh` | Launches the Flask frontend web app (`src/frontend/app.py`). |
| `train_cnn.sh` | Trains the CNN image-processing model. |
| `train_mlp.sh` | Trains the MLP tabular-processing model. |
| `train_fusion.sh` | Trains the multimodal fusion model. |
| `train_mlp_fusion.sh` | Trains the MLP-based fusion model. |
| `test_cnn.sh` | Evaluates the trained CNN model. |
| `test_mlp.sh` | Evaluates the trained MLP model. |
| `test_fusion.sh` | Evaluates the trained fusion model. |
| `test_mlp_fusion.sh` | Evaluates the trained MLP fusion model. |

**Design Structure:**
- Since the Handwriting Sample Consists of two different types of data (Image Data and Tabular Data), they are split it into two separate models
- Each section, Image Processing and Tabular Processing, are completely disjoint from each other and only come together in a multimodal fusion model. This way there can be optionally an image input or a CSV input, or both.
- There is a CNN to process the incoming Image Sample, and Grad-CAM is looped in during the convolution stage to create a heat map of the image to explain the model's decisions
- There is a MLP to process the Tabular Sample from the CSV
- Both are fed into a fusion model which uses there output to produce a combined output and more accurate diagnosis
- Results are processed and displayed in a GUI. In the case an Image Sample was used, then the Grad-CAM output will also be displayed in the GUI

**AI Disclosure:**
We used AI tools to facilate the following:
- The frontend was completely designed with AI
- Altough model architecting was mainly our design, AI tools helped debug issues such as transform sizing of images for the CNN and how to properly
freeze model layers for the Intermediate Fusion Model
- Gather diagnostics for each model. We used AI to automate test scripts as well as gather diagnostics like F1 score, precision, and recall during evaluation
- AI was used to debug issues with overfitting that occurred during training of the CNN
