import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import Optional, Dict, List

class LlamaDataset(Dataset):
    def __init__(
        self,
        split: str = 'train',
        data_path: str = 'data',
        max_seq_len: int = 2048,
        tokenizer_name: str = 'meta-llama/Llama-2-7b-hf'
    ):
        self.split = split
        self.data_path = data_path
        self.max_seq_len = max_seq_len
        
        # Tokenizer'ı yükle
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Özel tokenleri ekle
        special_tokens = {
            'think_token': '<|think|>',
            'end_think_token': '<|endthink|>',
            'answer_token': '<|answer|>',
            'end_answer_token': '<|endanswer|>'
        }
        self.tokenizer.add_special_tokens({'additional_special_tokens': list(special_tokens.values())})
        
        # Veriyi yükle
        self.data = self.load_data()
        
    def load_data(self) -> List[Dict]:
        file_path = f'{self.data_path}/{self.split}.json'
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    
    def prepare_input(
        self,
        text: str,
        target: Optional[str] = None,
        language: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:
        # Girdi metnini tokenize et
        inputs = self.tokenizer(
            text,
            max_length=self.max_seq_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Target varsa onu da tokenize et
        if target is not None:
            target_inputs = self.tokenizer(
                target,
                max_length=self.max_seq_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            inputs['target'] = target_inputs['input_ids'].squeeze(0)
        
        # Dil bilgisini ekle
        if language is not None:
            inputs['language'] = language
            
        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            **({} if target is None else {'target': inputs['target']}),
            **({} if language is None else {'language': language})
        }
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        
        # Veriyi hazırla
        return self.prepare_input(
            text=item['input'],
            target=item.get('target'),
            language=item.get('language')
        )

def create_attention_mask(
    seq_len: int,
    causal: bool = True,
    padding_mask: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Attention mask oluştur"""
    if causal:
        # Causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        mask = ~mask
    else:
        # Bidirectional mask
        mask = torch.ones(seq_len, seq_len).bool()
    
    # Padding mask varsa uygula
    if padding_mask is not None:
        batch_size = padding_mask.size(0)
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
        padding_mask = padding_mask.unsqueeze(1).expand(-1, seq_len, -1)
        mask = mask & padding_mask
    
    return mask

def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Batch'leri birleştir"""
    # Batch'deki tüm öğeleri topla
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    
    # Target varsa onları da topla
    targets = None
    if 'target' in batch[0]:
        targets = torch.stack([item['target'] for item in batch])
    
    # Dil bilgisi varsa topla
    languages = None
    if 'language' in batch[0]:
        languages = [item['language'] for item in batch]
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        **({} if targets is None else {'target': targets}),
        **({} if languages is None else {'language': languages})
    } 