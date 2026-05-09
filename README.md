# DDH_Project-
DislocPred: An AI-powered diagnostic assistant utilizing YOLOv11-pose to automatically calculate the Acetabular Index (AI) and detect Developmental Dysplasia of the Hip (DDH) in pediatric X-rays
# DislocPred: AI Diagnostic Assistant for DDH

This repository contains the source code and deployment files for **DislocPred**, an automated web-based diagnostic assistant designed to detect Developmental Dysplasia of the Hip (DDH) in pediatric pelvic X-rays.

## Live Demo
A live working demo of the system is deployed on Hugging Face Spaces. You can test it here: 
**[Insert Your Hugging Face Space Link Here]**

## Project Overview
Diagnosing DDH relies heavily on manually calculating the Acetabular Index (AI) angle from X-rays, a process that is time-consuming and prone to inter-observer variability. DislocPred automates this workflow. It extracts the necessary anatomical landmarks using computer vision and calculates the exact joint angles programmatically, acting as an objective "second opinion" for non-specialized pediatricians and clinics.

The model was fine-tuned on a highly curated dataset of pediatric radiographs, with ground-truth keypoints manually annotated by a team of expert doctors and validated by orthopedic surgeons.

## System Pipeline
The architecture avoids the "black box" AI approach by decoupling the landmark detection from the actual medical diagnosis. The pipeline operates in four main steps:

1. **Split Inference Strategy:** The input X-ray is cropped into isolated right and left hemispheres to prevent the model from confusing bilateral symmetry.
2. **Keypoint Extraction:** A lightweight, fine-tuned `YOLOv11n-pose` model predicts the exact (x, y) coordinates of the Triradiate Cartilages and Acetabular Rims.
3. **Geometric Engine:** The predicted coordinates are passed to a deterministic Python script. This engine constructs Hilgenreiner’s line and the acetabular roof lines using the slope method to calculate the final AI angle.
4. **Clinical Output:** The Gradio web interface displays the original image overlaid with the diagnostic lines and the calculated angles, ensuring full explainability.

## Repository Structure
* `app.py`: The main application file containing the Gradio UI setup, image processing pipeline, and the geometric math engine.
* `best.pt`: The custom-trained YOLOv11n-pose model weights.
* `requirements.txt`: The required Python dependencies to run the system.

## Local Installation & Setup
To run this project locally on your machine, ensure you have Python 3.10 or higher installed.

1. **Clone the repository:**
```bash
git clone [https://github.com/your-username/DislocPred.git](https://github.com/your-username/DislocPred.git)
cd DislocPred
