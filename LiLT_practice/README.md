# LiLT: A better language agnostic LayoutLM model
The script is based on this [tutorial](https://www.philschmid.de/fine-tuning-lilt)


# Setup Development Environment
```
# ubuntu
> sudo apt install -y tesseract-ocr
# python
> pip install -r requirements.txt --upgrade
# install git-fls for pushing model and logs to the hugging face hub
> sudo apt-get install git-lfs --yes
```

# Run a training script
```
> python3 train.py
```


# Run a inference script
```
> python3 inference.py
```