
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding
import torch
from tqdm import tqdm

class NIEstimator():
    
    def __init__(self, model, tokenizer) -> None:
        self.model = model.to("cuda")
        self.tokenizer = tokenizer
    
    def information_from_generator(self, generator, context_tokens=0):
        dataset = Dataset.from_generator(generator)
        dataset = dataset.map(lambda sample: self.tokenizer(sample["text"]),
                              remove_columns=["text"])

        # seems to be more stable if its one at a time
        
        dl = torch.utils.data.DataLoader(dataset,
                                         batch_size=1, 
                                         collate_fn=DataCollatorWithPadding(self.tokenizer, return_tensors="pt"),
                                         pin_memory=True)
        
        with torch.no_grad():
            for b_sample in tqdm(dl):
                b_id = b_sample.pop("id")
                logits = self.model(**b_sample.to("cuda")).logits[:,context_tokens:-1,:] # skip last
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                    
                target_ids = b_sample.input_ids[:,context_tokens+1:, None].long() # skip first + context  
                log_target_probs = torch.gather(log_probs, -1, target_ids).squeeze(-1).sum(axis=-1)
                
                yield {
                    "id": b_id[0],
                    "information" : -log_target_probs[0].cpu().item(),
                    "seq_len": b_sample.input_ids.shape[-1] - context_tokens+1,
                }

class HFNIEstimator(NIEstimator):
    def __init__(self, checkpoint_name):
        model = AutoModelForCausalLM.from_pretrained(checkpoint_name)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_name, 
                                                  padding_side="left")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if tokenizer.eos_token is None and tokenizer.pad_token is None:
            raise RuntimeError("No avaialble padding token")

        super().__init__(model=model, tokenizer=tokenizer)