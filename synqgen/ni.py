
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding
import torch
from tqdm import tqdm

class NIEstimator():
    
    def __init__(self, model, tokenizer) -> None:
        self.model = model.to("cuda")
        self.tokenizer = tokenizer
    
    def information_from_generator(self, 
                                   generator, 
                                   context_percentage=0,
                                   context_tokens=0):
        
        if context_percentage>0 and context_tokens>0:
            print("WARNING: Note that context_percentage and context_tokens are both defined we will use the value of context_percentage.")
        
        dataset = Dataset.from_generator(generator)
        dataset = dataset.map(lambda sample: self.tokenizer(sample["text"], truncation=True),
                              remove_columns=["text"])

        # seems to be more stable if its one at a time
        
        dl = torch.utils.data.DataLoader(dataset,
                                         batch_size=1, 
                                         collate_fn=DataCollatorWithPadding(self.tokenizer, return_tensors="pt"),
                                         pin_memory=True)
        
        with torch.no_grad():
            for b_sample in tqdm(dl):
                b_id = b_sample.pop("id")
                
                # dynamic context
                if context_percentage>0:
                    context_tokens = int(b_sample.input_ids.shape[-1]*context_percentage)
                
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
    def __init__(self, checkpoint_name, cache_dir=None, model_kwargs={}):
        model = AutoModelForCausalLM.from_pretrained(checkpoint_name, 
                                                     cache_dir = cache_dir,
                                                     **model_kwargs)
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_name, 
                                                  cache_dir = cache_dir,
                                                  padding_side="left")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if tokenizer.eos_token is None and tokenizer.pad_token is None:
            raise RuntimeError("No avaialble padding token")

        super().__init__(model=model, tokenizer=tokenizer)