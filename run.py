import gradio as gr
from transformers import pipeline, AutoModelForSeq2SeqLM
import pandas as pd
import os


def load_model(model_dir=None):
    try:
        if not model_dir:
            #get latest from models dir
            model_dir = os.path.join(os.getcwd(), "models")
            model_dir = os.path.join(model_dir, os.listdir(model_dir)[-1])
            model_dir = os.path.join(model_dir, os.listdir(model_dir)[-1])
        
        print(f"Using model from: {model_dir}\n")
        summarizer = pipeline("summarization", model=model_dir)
    except:
        print("Loading default t5-small!")
        summarizer = pipeline("summarization", 
                              model=AutoModelForSeq2SeqLM.from_pretrained("t5-small"), 
                              tokenizer=AutoTokenizer.from_pretrained("t5-small"))
    
    return summarizer

def load_examples(file=os.path.join(os.getcwd(), "examples.csv"), col="article", num=10):
    print(f"Loading {num} examples from: {file}")
    dt = pd.read_csv(file)
    dt = dt[col].iloc[:num]
    return dt.values


def summarize(text, prefix="summarize: "):
    text = prefix + text

    result = summarizer(text)
    return result[0]["summary_text"]


if __name__=="__main__":
    examples = [example for example in load_examples()]

    summarizer = load_model()

    demo = gr.Interface(
        fn=summarize,
        inputs=gr.inputs.Textbox(lines=5, label="Input Text"),
        outputs=gr.outputs.Textbox(label="Summarized text"),
        examples=examples,
        
    )

    demo.launch()