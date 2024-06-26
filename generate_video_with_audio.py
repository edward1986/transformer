# generate_video_with_audio.py

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np
from gtts import gTTS
from moviepy.editor import ImageSequenceClip, AudioFileClip, concatenate_videoclips
import torch

def generate_text_descriptions():
    model_name = 'gpt2'
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    input_text = "Create a sequence of images describing a sunrise over a mountain"
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    
    # Manually set the pad token id if it does not exist
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    attention_mask = (input_ids != tokenizer.pad_token_id).to(torch.long)

    # Use top_k sampling to generate multiple sequences
    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=50,
        num_return_sequences=5,
        do_sample=True,
        top_k=50,
        pad_token_id=tokenizer.eos_token_id
    )
    descriptions = [tokenizer.decode(o, skip_special_tokens=True) for o in output]

    return descriptions

def split_text_into_chunks(text, chunk_size=3):
    words = text.split()
    chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

def text_to_image(text, image_size=(800, 400), font_size=48):
    img = Image.new('RGB', image_size, color=(73, 109, 137))
    d = ImageDraw.Draw(img)
    
    # Use default PIL font with adjusted size
    font = ImageFont.load_default()
    text_bbox = d.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    scale_factor = font_size / 10  # Adjusting size manually since default font size is approximately 10
    text_width = int(text_width * scale_factor)
    text_height = int(text_height * scale_factor)
    text_x = (image_size[0] - text_width) // 2
    text_y = (image_size[1] - text_height) // 2

    # Draw scaled text
    d.text((text_x, text_y), text, fill=(255, 255, 0), font=font)

    return img

def generate_audio(text, filename):
    tts = gTTS(text)
    tts.save(filename)

def create_video_with_audio(images, audio_files, output_path='output_video.mp4'):
    clips = []
    for img, audio_file in zip(images, audio_files):
        img_array = np.array(img)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        img_clip = ImageSequenceClip([img_array], fps=1)
        audio_clip = AudioFileClip(audio_file)
        img_clip = img_clip.set_audio(audio_clip)
        clips.append(img_clip)
    
    final_clip = concatenate_videoclips(clips)
    final_clip.write_videofile(output_path, fps=1)

def main():
    descriptions = generate_text_descriptions()
    descriptions_chunks = [split_text_into_chunks(desc) for desc in descriptions]
    
    images = []
    audio_files = []
    
    for desc_chunks in descriptions_chunks:
        for chunk in desc_chunks:
            img = text_to_image(chunk)
            images.append(img)
            
            audio_filename = f"audio_{len(audio_files)+1}.mp3"
            generate_audio(chunk, audio_filename)
            audio_files.append(audio_filename)
    
    create_video_with_audio(images, audio_files)

if __name__ == "__main__":
    main()
