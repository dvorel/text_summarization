
class Tokenizer(object):
    def __init__(self, 
        tokenizer, 
        enc_max_len=512, 
        dec_max_len=128,
        padding="max_length", 
        truncate=True,
        prefix = "summarize: "
    ):
        self.padding = padding
        self.truncate = truncate    
        self.enc_max = enc_max_len
        self.dec_max = dec_max_len
        self.prefix = prefix
        self.tokenizer = tokenizer

        self.tokenizer.bos_token = self.tokenizer.cls_token
        self.tokenizer.eos_token = self.tokenizer.sep_token

    def __call__(self, sample):
        text = [sample["text"]]
        if self.prefix:
            text = [self.prefix + doc for doc in text]
        
        inputs = self.tokenizer(text, 
                                padding=self.padding, 
                                truncation=self.truncate, 
                                max_length=self.enc_max)

        outputs = self.tokenizer(sample["summary"], 
                                 padding=self.padding, 
                                 truncation=self.truncate, 
                                 max_length=self.enc_max)
        
        inputs["labels"] = outputs["input_ids"]

        return inputs


class ToTensor(object):
    """Convert ndarray to tensor"""

    def __call__(self, sample):
        txt, sum = sample["text"], sample["summary"]

        return {"text" : txt, "summary" : sum}