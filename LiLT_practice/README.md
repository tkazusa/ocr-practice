# LiLT: A better language agnostic LayoutLM model
The script is based on this [tutorial](https://www.philschmid.de/fine-tuning-lilt)


# Setup Environment
```
# ubuntu
> sudo apt install -y tesseract-ocr
# python
> pip install -r requirements.txt --upgrade
# install git-fls for pushing model and logs to the hugging face hub
> sudo apt-get install git-lfs --yes
```

In Huggingface `LayoutLMv3FeatureExtractor`, [Tesseract] is used as an OCR engine.
To utilize it for multiple languages, the language dataset files are neede.

```bash
> tesseract --list-langs
List of available languages (2):
eng
osd
> wget -P /usr/share/tesseract-ocr/4.00/tessdata/ https://github.com/tesseract-ocr/tessdata/raw/master/jpn.traineddata
> export TESSDATA_PREFIX /usr/share/tesseract-ocr/4.00/tessdata/
> tesseract --list-langs
List of available languages (3):
eng
jpn
osd
```
You can see [feature_extraction](./feature_extraction/) for more details.


# Run the training script of an entity segmentation model
```
> python3 train.py
```
You can see [train](./train/) for more details.

# Run a inference script
```
> python3 inference.py
```