
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding
import torch
from tqdm import tqdm
from dataclasses import dataclass
from typing import List
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union


@dataclass
class ConvertToTensor:

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        
        samples = {"input_ids": [], "attention_mask": [], "id":[]}
        
        for feature in features:
            samples["id"].append(feature["id"])
            samples["input_ids"].append(feature["input_ids"])
            samples["attention_mask"].append(feature["attention_mask"])
        
        samples["input_ids"] = torch.as_tensor(samples["input_ids"])
        samples["attention_mask"] = torch.as_tensor(samples["attention_mask"])
        
        return samples

@dataclass
class LMInput:
    input_ids: List[int]
    attention_mask: List[int]
    
    def __len__(self):
        return len(self.input_ids)

class MovingWindow:
    "https://stackoverflow.com/questions/64118654/best-way-to-implement-moving-window-in-python-for-loop"
    def __init__(self, tokens, window_size, step):
        self.current = -step
        self.last = len(tokens.input_ids) - window_size + 1
        self.remaining = (len(tokens.input_ids) - window_size) % step
        self.tokens = tokens
        self.window_size = window_size
        self.step = step

    def __iter__(self):
        return self

    def __next__(self):
        self.current += self.step
        if self.current < self.last:
            return LMInput(input_ids=self.tokens.input_ids[self.current : self.current + self.window_size],
                           attention_mask=self.tokens.attention_mask[self.current : self.current + self.window_size])
        elif self.remaining:
            self.remaining = 0
            return LMInput(input_ids=self.tokens.input_ids[-self.window_size:],
                           attention_mask=self.tokens.attention_mask[-self.window_size:])
        else:
            raise StopIteration
        
        
def sliding_window(tokenizer, window_size, step_size):
    def func(sample):
        samples = {"input_ids": [], "attention_mask": [], "id":[]}

        for i in range(len(sample["id"])):
            for j,s_sample in enumerate(MovingWindow(tokenizer(sample["text"][i]), window_size, step_size)):
                samples["input_ids"].append(s_sample.input_ids)
                samples["attention_mask"].append(s_sample.attention_mask)
                _id = sample["id"][i]
                samples["id"].append(f"{_id}_{j}")
        
        return samples
    return func
        
class NIEstimator():
    
    def __init__(self, model, tokenizer, cache_dir=None) -> None:
        self.model = model.to("cuda")
        self.tokenizer = tokenizer
        self.cache_dir = cache_dir
    
    def information_from_generator(self, 
                                   generator, 
                                   context_percentage=0,
                                   context_tokens=0,
                                   progress_bar=True):
        
        if context_percentage>0 and context_tokens>0:
            print("WARNING: Note that context_percentage and context_tokens are both defined we will use the value of context_percentage.")
        
        dataset = Dataset.from_generator(generator, cache_dir=self.cache_dir)
        #This needs to change to a sliding window
        _window_size = min(self.tokenizer.model_max_length, 4096)
        _step_size = _window_size//2
        
        sliding_window_f = sliding_window(tokenizer=self.tokenizer,
                                          window_size=_window_size,
                                          step_size=_step_size)
        
        dataset = dataset.map(sliding_window_f, batched=True, batch_size=8, remove_columns=["text"])

        # seems to be more stable if its one at a time
        print("New size of dataset", len(dataset))
        dl = torch.utils.data.DataLoader(dataset,
                                         batch_size=1, 
                                         collate_fn=ConvertToTensor(),
                                         pin_memory=True)
        
        if progress_bar:
            dl = tqdm(dl)
        
        with torch.no_grad():
            for b_sample in dl:
                b_id = b_sample.pop("id")
                
                input_ids = b_sample["input_ids"].to("cuda")
                
                # dynamic context
                if context_percentage>0:
                    context_tokens = int(input_ids.shape[-1]*context_percentage)
                
                logits = self.model(input_ids=input_ids,
                                    attention_mask=b_sample["attention_mask"].to("cuda")).logits[:,context_tokens:-1,:] # skip last
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                    
                target_ids = input_ids[:,context_tokens+1:, None].long() # skip first + context  
                log_target_probs = torch.gather(log_probs, -1, target_ids).squeeze(-1).sum(axis=-1)
                
                yield {
                    "id": b_id[0],
                    "information" : -log_target_probs[0].cpu().item(),
                    "seq_len": b_sample["input_ids"].shape[-1] - context_tokens+1,
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

        super().__init__(model=model, tokenizer=tokenizer, cache_dir=cache_dir)