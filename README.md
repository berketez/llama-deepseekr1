# llama-deepseekr1
# Self-Improving Llama: Reasoning Model Inspired by DeepSeek-R1

This project presents a self-improving language model inspired by the DeepSeek-R1 paper, built upon the Llama-3 architecture. The model is enhanced with Reinforcement Learning (RL) and Chain-of-Thought (CoT) mechanisms, enabling it to optimize reasoning processes dynamically and self-reflect for better performance on complex reasoning tasks.

## Model Architecture

Our model is built upon the Llama-3 architecture and incorporates the following key features:

*   **Llama-3 Based Core:** The core Transformer blocks and embedding layers are based on the Llama-3 model.
*   **YARN (Yet Another RoPE) Positional Encoding:** Employs the YARN mechanism for improved performance on long sequences.
*   **Dynamic Thinking:** The model dynamically adjusts its thinking depth and dropout rate during the Chain-of-Thought (CoT) process.
*   **Group Relative Policy Optimization (GRPO):** The model uses GRPO to evaluate rewards on a group basis, enabling a more efficient RL training process.
*   **Self-Reflection:** The model evaluates its own steps and results, re-evaluates when necessary, and improves itself iteratively.
*   **RL Feedback:** The model uses a reward and punishment system during training, enabling it to make better decisions and improve reasoning skills.

This model is inspired by the DeepSeek-R1 paper, utilizing the Llama-3 architecture as its foundation.

## Setup

1.  **Requirements:**
    Install the following libraries and versions:
    ```bash
    pip install -r requirements.txt
    ```
    The `requirements.txt` file should include the following libraries:
    ```
        torch==2.1.0
        transformers==4.35.2
        datasets==2.16.1
        tqdm==4.66.1
        accelerate==0.25.0
        sentencepiece==0.1.99
        langid==1.1.6
    ```
    The `langid` library is required for the `compute_language_consistency_reward` function. You can remove `langid` from `requirements.txt` if you choose not to use it.

2.  **Project Directory Structure:**
    ```
    deepseek_llama/
    ├── model.py         # Contains the model architecture files
    ├── train.py         # Contains the training scripts
    ├── dataset.py      # Contains the dataset creation code
    ├── requirements.txt   # Lists the library dependencies
    ├── README.md        # This file
    ├── LICENSE          # License information
    └── datasets/        # Example dataset files
       ├── train_data_1.txt
       └── train_data_2.txt
    ```
    The `datasets` directory contains example training data in the `.txt` format. You can add your own data to this directory in the `.txt` format.
    
3. **Dataset Formatting:**
  Your dataset should be in `.json` format with the following structure:
   ```json
   [
     {
       "input": "Question or explanation...",
       "target": "Target output (optional)",
       "language": "Input language (optional)"
     },
     ...
    ]


   Training
Training Command:
You can start training by using the following command and modify the parameters using argparse in train.py:
python train.py --num_gpus [number of gpus]  --batch_size [batch size] --learning_rate [learning rate] --epochs [number of epochs] --dim [model dimension] --n_layers [number of layers] --n_heads [number of heads] --vocab_size [vocabulary size] --max_seq_len [max sequence length]


num_gpus: The number of GPUs to use for training (e.g., python train.py --num_gpus 4).

batch_size: The batch size per GPU (e.g., python train.py --batch_size 32).

learning_rate: The learning rate (e.g., python train.py --learning_rate 3e-4).

epochs: The number of training epochs (e.g., python train.py --epochs 10).

dim: The hidden dimension of the model (e.g., python train.py --dim 4096).

n_layers: The number of transformer layers (e.g., python train.py --n_layers 32).

n_heads: The number of attention heads (e.g., python train.py --n_heads 32).

vocab_size: The vocabulary size (e.g., python train.py --vocab_size 32000).

max_seq_len: The maximum sequence length (e.g., python train.py --max_seq_len 2048).

Additional Parameters: You can also add the other parameters from train.py to the command line.

Note: You can change the model parameters by defining them in command line argumnents.

Training Process:

The model is trained using an iterative RL algorithm defined in train.py.

The model undergoes an evaluation step every epoch.

Checkpoints (.pt files) are saved regularly to the checkpoints directory.

Log information is saved to the logs directory.

Model Structure
Model File (model.py)
The main classes that make up the model architecture are as follows:

RMSNorm: Root Mean Square Layer Normalization layer.

RotaryEmbedding: YARN-based rotary positional encoding layer.

SelfAttention: Multi-Head Self-Attention layer.

FeedForward: MLP layer.

TransformerBlock: Transformer block that combines the attention and feed-forward layers.

Llama: Base Llama model class.

RLModule: Contains the value estimators and reward calculation mechanisms for Reinforcement Learning.

SelfImprovingLlama: The self-improving Llama model that incorporates RL and CoT features.

Training File (train.py)
The functions and loops that manage the training process are as follows:

setup: Sets up the distributed training.

train_step: Executes a single training step.

evaluate: Measures the evaluation performance of the model.

train: Runs the main training loop.

main: Parses training arguments and initiates the training process.


Dataset File (dataset.py)
Includes classes for data loading and preprocessing. The example contains the LlamaDataset class.

The LlamaDataset class automatically performs tokenization, padding and masking.

Usage
Loading the Model:
from model import SelfImprovingLlama
import torch

# Load the model
model = SelfImprovingLlama(
    vocab_size=[vocab_size],
    dim=[dim],
    depth=[depth],
    num_heads=[num_heads],
    mlp_ratio=[mlp_ratio],
    dropout=[dropout],
    attn_dropout=[attn_dropout]
)

# Load a checkpoint (if available)
checkpoint = torch.load('checkpoints/step_....pt')
model.load_state_dict(checkpoint['model_state_dict'])

Generating Output:
input_ids = torch.randint(0, [vocab_size], (1, [seq_len])).to(device)

# Generate output using CoT
output = model.generate_with_cot(input_ids, max_length=200)

# Generate improved output with RL
output = model.generate_with_improvement(input_ids, max_length=200)

License
This project is released under the MIT License.

Contributing
Contributions are welcome! Please feel free to submit bug reports, improvement suggestions, and pull requests.

Contact
For any questions, suggestions, or feedback, please contact us through this repository.
berketezgocen97@hotmail.com
