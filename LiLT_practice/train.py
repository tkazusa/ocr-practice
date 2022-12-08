from functools import partial

import evaluate
import numpy as np
from datasets import (Array2D, ClassLabel, Features, Sequence, Value,
                      load_dataset)
from huggingface_hub import HfFolder
from PIL import Image, ImageDraw, ImageFont
from torch import Tensor
from transformers import (AutoTokenizer, BatchEncoding,
                          LayoutLMv3FeatureExtractor, LayoutLMv3Processor,
                          LiltForTokenClassification, Trainer,
                          TrainingArguments)


# preprocess function to perpare into the correct format for the model
def process(sample, processor=None) -> BatchEncoding: 
    encoding = processor(
        sample["image"].convert("RGB"),
        sample["tokens"],
        boxes=sample["bboxes"],
        word_labels=sample["ner_tags"],
        padding="max_length",
        truncation=True,
    )
    # remove pixel values not needed for LiLT
    del encoding["pixel_values"]
    return encoding


# helper function to unnormalize bboxes for drawing onto the image
def unnormalize_box(bbox: int, width: int, height: int) -> list:
    return [
        width * (bbox[0] / 1000),
        height * (bbox[1] / 1000),
        width * (bbox[2] / 1000),
        height * (bbox[3] / 1000),
    ]


# draw results onto the image
def draw_boxes(image: Image, boxes: Tensor, predictions: Tensor) -> Image:
    width, height = image.size
    normalizes_boxes = [unnormalize_box(box, width, height) for box in boxes]

    # draw predictions over the image
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    for prediction, box in zip(predictions, normalizes_boxes):
        if prediction == "O":
            continue
        draw.rectangle(box, outline="black")
        draw.rectangle(box, outline="blue")
        # draw.text((box[0] + 10, box[1] - 10), text=prediction, fill=label2color[prediction], font=font)
    return image


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    all_predictions = []
    all_labels = []
    for prediction, label in zip(predictions, labels):
        for predicted_idx, label_idx in zip(prediction, label):
            if label_idx == -100:
                continue
            all_predictions.append(ner_labels[predicted_idx])
            all_labels.append(ner_labels[label_idx])
    return metric.compute(predictions=[all_predictions], references=[all_labels])


if __name__ == "__main__":
    #dataset_id ="nielsr/funsd"
    dataset_id ="nielsr/funsd-layoutlmv3"
    dataset = load_dataset(dataset_id)

    print(f"Train dataset size: {len(dataset['train'])}")
    print(f"Test dataset size: {len(dataset['test'])}")
    # Train dataset size: 149
    # Test dataset size: 50

    image = dataset['train'][34]['image']
    image = image.convert("RGB")
    labels = dataset['train'].features['ner_tags'].feature.names
    print(f"Available labels: {labels}")

    id2label = {v: k for v, k in enumerate(labels)}
    label2id = {k: v for v, k in enumerate(labels)}
    # Available labels: ['O', 'B-HEADER', 'I-HEADER', 'B-QUESTION', 'I-QUESTION', 'B-ANSWER', 'I-ANSWER']
    model_id="SCUT-DLVCLab/lilt-roberta-en-base"

    # use LayoutLMv3 processor without ocr since the dataset already includes the ocr text
    feature_extractor = LayoutLMv3FeatureExtractor(apply_ocr=False) # set
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # cannot use from_pretrained since the processor is not saved in the base model
    processor = LayoutLMv3Processor(feature_extractor, tokenizer)

    # we need to define custom features
    features = Features(
        {
            "input_ids": Sequence(feature=Value(dtype="int64")),
            "attention_mask": Sequence(feature=Value(dtype="int64")),
            "bbox": Array2D(dtype="int64", shape=(512, 4)),
            "labels": Sequence(ClassLabel(names=labels)),
        }
    )

    # process the dataset and format it to pytorch
    proc_dataset = dataset.map(
        partial(process, processor=processor),
        remove_columns=["image", "tokens", "ner_tags", "id", "bboxes"],
        features=features,
    ).with_format("torch")

    print(proc_dataset["train"].features.keys())
    # dict_keys(['input_ids', 'token_type_ids', 'attention_mask', 'bbox','lables'])

    bbox_34 = proc_dataset["train"]["bbox"][34]
    labels_34 = proc_dataset["train"]["labels"][34]

    label2color = {
        "B-HEADER": "blue",
        "B-QUESTION": "red",
        "B-ANSWER": "green",
        "I-HEADER": "blue",
        "I-QUESTION": "red",
        "I-ANSWER": "green",
    }

    # Check the procesed dataset
    image = draw_boxes(image, bbox_34, labels_34)
    image.resize((350,450))
    image.save('images/dataset_34_ocr.jpg')


    # huggingface hub model id
    model_id = "SCUT-DLVCLab/lilt-roberta-en-base"
    # load model with correct number of labels and mapping
    model = LiltForTokenClassification.from_pretrained(
        model_id, num_labels=len(labels), label2id=label2id, id2label=id2label
    )
    # load seqeval metric
    metric = evaluate.load("seqeval")
    # labels of the model
    ner_labels = list(model.config.id2label.values())

    # hugging face parameter
    repository_id = "lilt-en-funsd"

    # Define training args
    training_args = TrainingArguments(
        output_dir=repository_id,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        fp16=True,
        learning_rate=5e-5,
        max_steps=2500,
        # logging & evaluation strategies
        logging_dir=f"{repository_id}/logs",
        logging_strategy="steps",
        logging_steps=200,
        evaluation_strategy="steps",
        save_strategy="steps",
        save_steps=200,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="overall_f1",
        # push to hub parameters
        report_to="tensorboard",
        push_to_hub=True,
        hub_strategy="every_save",
        hub_model_id=repository_id,
        hub_token=HfFolder.get_token(),
    )

    # Create Trainer instance
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=proc_dataset["train"],
        eval_dataset=proc_dataset["test"],
        compute_metrics=compute_metrics,
    )

    trainer.train()
    

    # Evaluation 
    trainer.evaluate()

    # change apply_ocr to True to use the ocr text for inference
    processor.feature_extractor.apply_ocr = True

    # Save processor and create model card
    processor.save_pretrained(repository_id)
    trainer.create_model_card()
    trainer.push_to_hub()
