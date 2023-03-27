"""
Sources:
https://www.kaggle.com/datasets/gowrishankarp/newspaper-text-summarization-cnn-dailymail
https://huggingface.co/docs/transformers/tasks/summarization

"""
import os
import numpy as np

from datasets import load_dataset
import evaluate
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq, AutoTokenizer


def get_save_path(dir="models"):
    cwd = os.path.join(os.getcwd(), dir)
    os.makedirs(cwd, exist_ok=True)

    return os.path.join(cwd, str(len(os.listdir(cwd))+1))

def set_model_params(model, t):
    model.config.decoder_start_token_id = t.bos_token_id                                             
    model.config.eos_token_id = t.eos_token_id

    # sensible parameters for beam search
    # set decoding params                               
    model.config.max_length = 142
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4
    model.config.vocab_size = model.config.encoder.vocab_size

def compute_metrics(eval_pred, eval):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = eval.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}

def preprocess(examples, prefix= "summarize: "):
    inputs = [prefix + doc for doc in examples["article"]]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

    labels = tokenizer(text_target=examples["highlights"], max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]

    return model_inputs

if __name__=="__main__":
    PATH = "datasets/cnn_dailymail/"
    BATCH_SIZE = 16
    ENC_MAX = 512
    DEC_MAX = 128
    WORKERS = 0
    MODEL = "t5-small"


    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    dataset = load_dataset("csv", 
                           data_files={
                                "train": os.path.join(PATH, "train.csv"), 
                                "test": os.path.join(PATH, "test.csv")})

    tokenized_dataset = dataset.map(preprocess, batched=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

    rogue = evaluate.load("rouge")

    training_args = Seq2SeqTrainingArguments(
        output_dir=get_save_path(),
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        weight_decay=0.01,
        save_total_limit=5,
        num_train_epochs=4,
        predict_with_generate=True,
        fp16=True,
        push_to_hub=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
