from datasets import load_dataset
from PIL import Image, ImageDraw, ImageFont
from torch import Tensor
from transformers import LayoutLMv3Processor, LiltForTokenClassification


# helper function to unnormalize bboxes for drawing onto the image
def unnormalize_box(bbox: int, width: int, height: int) -> list:
    return [
        width * (bbox[0] / 1000),
        height * (bbox[1] / 1000),
        width * (bbox[2] / 1000),
        height * (bbox[3] / 1000),
    ]


# draw results onto the image
def draw_boxes(image: Image, boxes: Tensor, predictions: Tensor, label2color: dict) -> Image:
    width, height = image.size
    normalizes_boxes = [unnormalize_box(box, width, height) for box in boxes]

    # draw predictions over the image
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for prediction, box in zip(predictions, normalizes_boxes):
        if prediction == "O":
            continue
        draw.rectangle(box, outline="black")
        draw.rectangle(box, outline=label2color[prediction])
        draw.text((box[0] + 10, box[1] - 10), text=prediction, fill=label2color[prediction], font=font)
    return image



# run inference
def run_inference(image, model, processor, label2color, output_image=True):
    # create model input
    encoding = processor(image, return_tensors="pt")
    del encoding["pixel_values"]
    # run inference
    outputs = model(**encoding)
    predictions = outputs.logits.argmax(-1).squeeze().tolist()
    # get labels
    labels = [model.config.id2label[prediction] for prediction in predictions]
    if output_image:
        return draw_boxes(image, encoding["bbox"][0], labels, label2color)
    else:
        return labels

if __name__ == "__main__":

    dataset_id ="nielsr/funsd-layoutlmv3"
    dataset = load_dataset(dataset_id)

    # load model and processor from huggingface hub
    model = LiltForTokenClassification.from_pretrained("philschmid/lilt-en-funsd")
    processor = LayoutLMv3Processor.from_pretrained("philschmid/lilt-en-funsd")

    label2color = {
        "B-HEADER": "blue",
        "B-QUESTION": "red",
        "B-ANSWER": "green",
        "I-HEADER": "blue",
        "I-QUESTION": "red",
        "I-ANSWER": "green",
    }

    test_image = dataset["test"][34]["image"]
    output_image = run_inference(image=test_image, model=model, processor=processor, label2color=label2color)
    output_image.resize((350,450))
    output_image.save('images/dataset_34_inference.jpg')