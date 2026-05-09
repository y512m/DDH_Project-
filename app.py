import gradio as gr
import cv2
import numpy as np
import math
from ultralytics import YOLO

# Load Model
MODEL_PATH = "best.pt"
model = YOLO(MODEL_PATH, task="pose")

def get_hip_points(image_crop, model, conf=0.25):
    results = model(image_crop, conf=conf, verbose=False)
    for r in results:
        if r.keypoints and r.keypoints.shape[1] >= 2:
            kpts = r.keypoints.xy[0].cpu().numpy().astype(int)
            valid_kpts = [p for p in kpts if p[0] != 0 and p[1] != 0]
            if len(valid_kpts) >= 2:
                valid_kpts.sort(key=lambda p: p[1])
                return valid_kpts[-1], valid_kpts[0]
    return None, None

def calculate_slope_angle(center, rim, other_center):
    try:
        if other_center[0] - center[0] == 0: m_h = 0
        else: m_h = (other_center[1] - center[1]) / (other_center[0] - center[0])
        if rim[0] - center[0] == 0: m_r = 1e9
        else: m_r = (rim[1] - center[1]) / (rim[0] - center[0])
        tan_theta = abs((m_r - m_h) / (1 + m_h * m_r))
        return math.degrees(math.atan(tan_theta))
    except: return 0.0

def get_diagnosis_style(angle):
    if angle >= 30: return "Dysplasia (High Severity)", "#FF4B4B"
    if angle >= 25: return "Borderline / Mild", "#FFA500"
    return "Normal", "#09AB3B"

def analyze_xray(input_img, conf):
    image = np.array(input_img)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    h, w, _ = image.shape
    mid_x = w // 2
    
    right_hip_img = image[:, :mid_x]
    left_hip_img = image[:, mid_x:]
    
    r_tri, r_rim = get_hip_points(right_hip_img, model, conf)
    l_tri_local, l_rim_local = get_hip_points(left_hip_img, model, conf)
    
    if r_tri is None or l_tri_local is None:
        return image, "⚠️ **Error:** Could not detect anatomical landmarks. Try adjusting the confidence threshold."

    l_tri_global = (l_tri_local[0] + mid_x, l_tri_local[1])
    l_rim_global = (l_rim_local[0] + mid_x, l_rim_local[1])
    
    # Draw Hilgenreiner's line (Base Line)
    cv2.line(image, tuple(r_tri), tuple(l_tri_global), (0, 255, 255), 3)
    
    ang_r = calculate_slope_angle(r_tri, r_rim, l_tri_global)
    diag_r, color_r = get_diagnosis_style(ang_r)
    
    ang_l = calculate_slope_angle(l_tri_global, l_rim_global, r_tri)
    diag_l, color_l = get_diagnosis_style(ang_l)
    
    # Formatting the output text with Markdown
    result_text = f"### 🩺 Clinical Assessment Results\n\n"
    result_text += f"**Right Hip AI:** `{ang_r:.1f}°` ➔ **{diag_r}**\n\n"
    result_text += f"**Left Hip AI:** `{ang_l:.1f}°` ➔ **{diag_l}**"

    # Draw Roof Lines
    cv2.line(image, tuple(r_tri), tuple(r_rim), (0, 0, 255), 3)
    cv2.line(image, tuple(l_tri_global), tuple(l_rim_global), (0, 0, 255), 3)
    
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), result_text

# ==========================================
# Gradio UI Design (Enhanced & English)
# ==========================================
custom_theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="slate",
)

with gr.Blocks(title="DislocPred: DDH System", theme=custom_theme) as demo:
    
    # Header Section
    with gr.Row():
        gr.Markdown(
            """
            # 🏥 DislocPred: AI Diagnostic Assistant
            **Automated Detection & Grading for Developmental Dysplasia of the Hip (DDH)** 
            Upload a pediatric pelvic X-ray (AP view). The system will analyze the anatomical landmarks, calculate the Acetabular Index (AI), and provide a clinical evaluation.
            ### 👨‍💻 Developed by : Yousef Hamdan
            """
        )
    
    # Main Application Area
    with gr.Row():
        # Left Column: Inputs
        with gr.Column(scale=1):
            gr.Markdown("### 📥 Input Panel")
            input_img = gr.Image(type="pil", label="Upload X-ray Image")
            
            with gr.Accordion("⚙️ Advanced Model Settings", open=False):
                conf_slider = gr.Slider(0.1, 1.0, value=0.25, step=0.05, label="Detection Confidence Threshold")
                
            btn = gr.Button("🔍 Run AI Analysis", variant="primary")
            
        # Right Column: Outputs
        with gr.Column(scale=1):
            gr.Markdown("### 📤 Diagnostic Output")
            output_img = gr.Image(label="Annotated Radiograph")
            output_text = gr.Markdown(label="Results")

    # Connect logic to UI
    btn.click(
        fn=analyze_xray, 
        inputs=[input_img, conf_slider], 
        outputs=[output_img, output_text]
    )

# Launch with proper Hugging Face settings
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)