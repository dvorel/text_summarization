# text_summarization
Text summarization using pretrained t5-small transformer.

Summarization is achived using finetuned t5-small transformer model and auto tokenizer from huggingface transformers python module. 

## How to use:
    1. Create Environment with requirements.txt
    2. Execute run.py script and open localhost:7860

## Fine Tune:
    1. Create Environment with requirements.txt
    2. Download or create dataset for model finetuning
        -csv should contain: 
            -id
            -input
            -output
    3. Edit:
        -PATH to match your model path
        -BATCH_SIZE to match your hardware capabilities
    4. After training is done, execute run.py and test your finetuned model

## App:

https://user-images.githubusercontent.com/24752476/228027212-e5834a0d-d0d3-4938-9340-122d6646cc8c.mp4
