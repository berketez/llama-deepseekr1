import os
import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from dataclasses import dataclass
from model import ModelArgs, SelfImprovingLlama
from dataset import LlamaDataset

@dataclass
class TrainingArgs:
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    epochs: int = 10
    warmup_steps: int = 2000
    gradient_clip: float = 1.0
    save_every: int = 1000
    eval_every: int = 500
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train_step(model, batch, optimizer, args, device):
    model.train()
    optimizer.zero_grad()
    
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch.get('attention_mask', None)
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    
    outputs = model.forward_with_rl(
        x=input_ids,
        mask=attention_mask,
        target=batch.get('target', None),
        input_language=batch.get('language', None)
    )
    
    loss = outputs['policy_loss']
    loss.backward()
    
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.gradient_clip)
    optimizer.step()
    
    return {
        'loss': loss.item(),
        'reward': outputs['rewards']['total_reward'].mean().item()
    }

def evaluate(model, eval_loader, device):
    model.eval()
    total_loss = 0
    total_reward = 0
    steps = 0
    
    with torch.no_grad():
        for batch in eval_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch.get('attention_mask', None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
                
            outputs = model.forward_with_rl(
                x=input_ids,
                mask=attention_mask,
                target=batch.get('target', None),
                input_language=batch.get('language', None)
            )
            
            total_loss += outputs['policy_loss'].item()
            total_reward += outputs['rewards']['total_reward'].mean().item()
            steps += 1
    
    return {
        'loss': total_loss / steps,
        'reward': total_reward / steps
    }

def train(rank, world_size, model_args, train_args):
    setup(rank, world_size)
    device = torch.device(f'cuda:{rank}')
    
    # Model oluştur
    model = SelfImprovingLlama(**vars(model_args)).to(device)
    model = DDP(model, device_ids=[rank])
    
    # Optimizer ve scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=train_args.learning_rate,
        weight_decay=train_args.weight_decay
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=train_args.epochs,
        eta_min=1e-5
    )
    
    # Veri yükleyicileri
    train_dataset = LlamaDataset(split='train')
    eval_dataset = LlamaDataset(split='eval')
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_args.batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=train_args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Checkpoint dizini oluştur
    if rank == 0:
        os.makedirs(train_args.checkpoint_dir, exist_ok=True)
        os.makedirs(train_args.log_dir, exist_ok=True)
    
    # Eğitim döngüsü
    global_step = 0
    for epoch in range(train_args.epochs):
        train_sampler.set_epoch(epoch)
        
        for batch in train_loader:
            metrics = train_step(model, batch, optimizer, train_args, device)
            global_step += 1
            
            if rank == 0 and global_step % train_args.save_every == 0:
                # Model kaydet
                checkpoint = {
                    'model_state_dict': model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'epoch': epoch,
                    'global_step': global_step
                }
                torch.save(
                    checkpoint,
                    os.path.join(train_args.checkpoint_dir, f'step_{global_step}.pt')
                )
            
            if rank == 0 and global_step % train_args.eval_every == 0:
                # Değerlendirme yap
                eval_metrics = evaluate(model, eval_loader, device)
                print(f'Step {global_step} | Train Loss: {metrics["loss"]:.4f} | '
                      f'Train Reward: {metrics["reward"]:.4f} | '
                      f'Eval Loss: {eval_metrics["loss"]:.4f} | '
                      f'Eval Reward: {eval_metrics["reward"]:.4f}')
        
        scheduler.step()
    
    cleanup()

def main():
    parser = argparse.ArgumentParser()
    # Model argümanları
    parser.add_argument('--dim', type=int, default=4096)
    parser.add_argument('--n_layers', type=int, default=32)
    parser.add_argument('--n_heads', type=int, default=32)
    parser.add_argument('--vocab_size', type=int, default=32000)
    parser.add_argument('--max_seq_len', type=int, default=2048)
    
    # Eğitim argümanları
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--num_gpus', type=int, default=torch.cuda.device_count())
    
    args = parser.parse_args()
    
    # Model ve eğitim argümanlarını oluştur
    model_args = ModelArgs(
        dim=args.dim,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        vocab_size=args.vocab_size,
        max_seq_len=args.max_seq_len
    )
    
    train_args = TrainingArgs(
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        epochs=args.epochs
    )
    
    # Distributed training başlat
    world_size = args.num_gpus
    mp.spawn(
        train,
        args=(world_size, model_args, train_args),
        nprocs=world_size,
        join=True
    )

if __name__ == "__main__":
    main() 