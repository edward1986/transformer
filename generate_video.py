from transformers import GPT2LMHeadModel, GPT2Tokenizer
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np

def generate_text_descriptions():
    model_name = 'gpt2'
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    input_text = "Create a sequence of images describing a sunrise over a mountain"
    input_ids = tokenizer.encode(input_text, return_tensors='pt')

    output = model.generate(input_ids, max_length=100, num_return_sequences=5)
    descriptions = [tokenizer.decode(o, skip_special_tokens=True) for o in output]

    return descriptions

def text_to_image(text, image_size=(800, 400), font_size=24):
    img = Image.new('RGB', image_size, color=(73, 109, 137))
    d = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    text_position = (10, 10)
    d.text(text_position, text, fill=(255, 255, 0), font=font)

    return img

def create_video(images, output_path='output_video.avi'):
    frame_width, frame_height = images[0].size
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'DIVX'), 1, (frame_width, frame_height))

    for img in images:
        img_array = np.array(img)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        out.write(img_array)

    out.release()

def main():
    descriptions = generate_text_descriptions()
    images = [text_to_image(desc) for desc in descriptions]
    
    for i, img in enumerate(images):
        img.save(f'image_{i+1}.png')
    
    create_video(images)

if __name__ == "__main__":
    main()
