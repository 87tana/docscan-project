from datasets import load_from_disk
from PIL import Image
import easyocr
from transformers import RobertaTokenizer, ViTImageProcessor, VisionEncoderDecoderModel

# Load everything fresh
ds = load_from_disk("data/test")
sample_hw = ds[1000]
image = sample_hw["image"].convert("RGB")
image.save("/home/tannaz/Documents/Projects/Docscan_project/test_handwritten_1000.png")

reader = easyocr.Reader(['en'])
results = reader.readtext("/home/tannaz/Documents/Projects/Docscan_project/test_handwritten_1000.png")

tokenizer = RobertaTokenizer.from_pretrained("microsoft/trocr-base-handwritten")
feature_extractor = ViTImageProcessor.from_pretrained("microsoft/trocr-base-handwritten")
trocr_model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

line_texts = []

for (bbox, text, confidence) in results:
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    left, right = min(xs), max(xs)
    top, bottom = min(ys), max(ys)

    line_crop = image.crop((left, top, right, bottom))

    pixel_values = feature_extractor(images=line_crop, return_tensors="pt").pixel_values
    generated_ids = trocr_model.generate(pixel_values, max_new_tokens=50)
    trocr_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    line_texts.append(trocr_text)

full_text = " ".join(line_texts)
print(full_text)
