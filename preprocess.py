import os
import typer
import cv2
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoFeatureExtractor, AutoModelForCausalLM, AutoTokenizer
from data.dataset import Sample
import yaml

app = typer.Typer()

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def extract_reference_image(video_path: str, output_path: str):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(output_path, frame)
    cap.release()

def generate_pose_sequence(video_path: str, output_path: str, dwpose_extractor, dwpose_model):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()

    inputs = dwpose_extractor(images=frames, return_tensors="pt")
    with torch.no_grad():
        pose_sequence = dwpose_model(**inputs).last_hidden_state
    np.save(output_path, pose_sequence.cpu().numpy())

def load_video(video_path, num_frames=24):
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    cap.release()
    
    return np.array(frames)

def generate_caption(video_path, caption_model, caption_tokenizer):
    video = load_video(video_path)
    prompt = "Please describe this video in detail."
    
    inputs = caption_model.build_conversation_input_ids(
        tokenizer=caption_tokenizer,
        query=prompt,
        images=[video],
        history=[],
    )
    
    inputs = {k: v.unsqueeze(0).to(caption_model.device) if isinstance(v, torch.Tensor) else [[v.to(caption_model.device)]] for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = caption_model.generate(**inputs, max_new_tokens=100)
        caption = caption_tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return caption

def create_text_prompt(video_path: str, output_path: str, caption_model, caption_tokenizer):
    caption = generate_caption(video_path, caption_model, caption_tokenizer)
    with open(output_path, 'w') as f:
        f.write(caption)

@app.command()
def preprocess(data_dir: str, config_path: str = "config/config.yaml"):
    config = load_config(config_path)
    
    # Initialize DWPose model and feature extractor
    dwpose_extractor = AutoFeatureExtractor.from_pretrained(config['model']['dwpose_path'])
    dwpose_model = AutoModelForCausalLM.from_pretrained(config['model']['dwpose_path'])
    dwpose_model.eval()
    
    # Initialize CogVLM2-Llama3-Caption model
    caption_tokenizer = AutoTokenizer.from_pretrained(config['model']['caption_model_path'])
    caption_model = AutoModelForCausalLM.from_pretrained(config['model']['caption_model_path'], trust_remote_code=True)
    caption_model.eval()
    
    if torch.cuda.is_available():
        dwpose_model = dwpose_model.cuda()
        caption_model = caption_model.cuda()

    samples = []
    for sample_dir in os.listdir(data_dir):
        sample_path = os.path.join(data_dir, sample_dir)
        if os.path.isdir(sample_path):
            driving_video = os.path.join(sample_path, 'driving.mp4')
            if os.path.exists(driving_video):
                reference_image = os.path.join(sample_path, 'reference.png')
                pose_sequence = os.path.join(sample_path, 'pose.npy')
                text_prompt = os.path.join(sample_path, 'prompt.txt')
                samples.append(Sample(reference_image, driving_video, pose_sequence, text_prompt))

    for sample in tqdm(samples, desc="Preprocessing dataset"):
        if not os.path.exists(sample.reference_image):
            extract_reference_image(sample.driving_video, sample.reference_image)
        if not os.path.exists(sample.pose_sequence):
            generate_pose_sequence(sample.driving_video, sample.pose_sequence, dwpose_extractor, dwpose_model)
        if not os.path.exists(sample.text_prompt):
            create_text_prompt(sample.driving_video, sample.text_prompt, caption_model, caption_tokenizer)

    print(f"Preprocessing completed for {len(samples)} samples.")

if __name__ == "__main__":
    app()
