import os
import math
from tqdm import tqdm
import torch
from torch import nn
from torch.nn import functional as F

from pathlib import Path
from torch.utils.data import DataLoader
from tokenizers import ByteLevelBPETokenizer
from datasets import Dataset, DatasetDict
from datasets import load_dataset, load_from_disk


local_path = Path("wikitext-103-raw-v1")


# 下载百科文章数据集 wikitext-103-raw-v1，或者从本地加载（如果之前已经下载过）
if local_path.exists():
    print("从本地读取数据集")
    ds = load_from_disk(str(local_path))
else:
    print("本地不存在，开始从 Hugging Face 下载")
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
    ds.save_to_disk(str(local_path))

print(ds)
print(ds["train"][1])
print(ds["train"][3])


# 将原始数据集中的文本进行清洗，并保存为纯文本，供后续训练使用。
out_dir = Path("data/wikitext103_clean_txt")

def normalize_line(text: str) -> str:
    """
    对单行 WikiText 做轻量清洗。
    注意：不删除 = Title = 这类标题结构。
    """
    text = text.rstrip("\n")
    text = text.rstrip()
    return text


def save_split_to_txt(split_name: str):
    out_file = out_dir / f"{split_name}.txt"

    with out_file.open("w", encoding="utf-8") as f:
        previous_blank = False

        for ex in ds[split_name]:
            line = normalize_line(ex["text"])

            if line.strip() == "":
                if not previous_blank:
                    f.write("\n")
                    previous_blank = True
                continue

            f.write(line + "\n")
            previous_blank = False

    print(f"saved: {out_file}")


if out_dir.exists():
    print(f"目录 {out_dir} 已存在，跳过保存")
else:
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "validation", "test"]:
        save_split_to_txt(split)


# 训练一个 Byte-Pair Encoding (BPE) 分词器，基于训练集文本构建词表。
paths = [
    "data/wikitext103_clean_txt/train.txt"
]

tokenizer = ByteLevelBPETokenizer()

tokenizer.train(
    files=paths,
    vocab_size=32000,
    min_frequency=2,
    special_tokens=[
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
    ],
)

out_dir = Path("tokenizer-wikitext103-bpe")
out_dir.mkdir(parents=True, exist_ok=True)

tokenizer.save_model(str(out_dir))

print("tokenizer saved")


# 训练集的数据切分
block_size = 512

files = {
    "train": "data/wikitext103_clean_txt/train.txt",
    "validation": "data/wikitext103_clean_txt/validation.txt",
    "test": "data/wikitext103_clean_txt/test.txt",
}

tokenizer = ByteLevelBPETokenizer(
    "tokenizer-wikitext103-bpe/vocab.json",
    "tokenizer-wikitext103-bpe/merges.txt",
)


def file_to_blocks(path):
    text = Path(path).read_text(encoding="utf-8")

    ids = tokenizer.encode(text).ids

    # 因为 labels 需要向后取 1 个 token，
    # 所以至少要预留 len(ids) - 1 的长度。
    total_length = len(ids) - 1

    # 截断到 block_size 的整数倍
    total_length = (total_length // block_size) * block_size

    input_ids = [
        ids[i:i + block_size]
        for i in range(0, total_length, block_size)
    ]

    labels = [
        ids[i + 1:i + block_size + 1]
        for i in range(0, total_length, block_size)
    ]

    return Dataset.from_dict({
        "input_ids": input_ids,
        "labels": labels,
    })


save_path = Path(f"data/wikitext103_bpe_lm_block{block_size}")

if save_path.exists():
    print(f"切片数据已存在，直接加载：{save_path}")
    lm_ds = load_from_disk(str(save_path))
else:
    print("切片数据不存在，开始生成...")

    lm_ds = DatasetDict({
        split: file_to_blocks(path)
        for split, path in files.items()
    })

    lm_ds.save_to_disk(str(save_path))
    print(f"切片数据已保存：{save_path}")

print(lm_ds)
print(lm_ds["train"][0].keys())
print(len(lm_ds["train"][0]["input_ids"]))


# 将数据集格式设置为 PyTorch tensors，方便后续 DataLoader 加载。
lm_ds.set_format(type="torch", columns=["input_ids", "labels"])

batch_size = 16

# 创建 DataLoader。
train_loader = DataLoader(
    lm_ds["train"],
    batch_size=batch_size,
    shuffle=True
)

val_loader = DataLoader(
    lm_ds["validation"],
    batch_size=batch_size,
    shuffle=False  # 验证集不需要打乱
)


# 语言模型的构建
class Head(nn.Module):
    """ 一个单头自注意力层 """
    def __init__(self, embed_dim, head_size):
        super().__init__()
        self.key = nn.Linear(embed_dim, head_size, bias=False)
        self.query = nn.Linear(embed_dim, head_size, bias=False)
        self.value = nn.Linear(embed_dim, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)    # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)

        # 计算注意力分数 (Affinity)
        wei = q @ k.transpose(-2, -1) * (k.shape[-1]**-0.5)  # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)

        # 融合 Value
        v = self.value(x)
        out = wei @ v # (B, T, head_size)
        return out


class MultiHeadAttention(nn.Module):
    """ 多个并行运行的自注意力头 """
    def __init__(self, embed_dim, num_heads, head_size):
        super().__init__()
        # 构建多头注意力模型
        # 提示: 使用 nn.ModuleList 存放多个独立的 Head
        self.heads = nn.ModuleList([Head(embed_dim, head_size) for _ in range(num_heads)])
        # 最后通过一个线性层投影回原始维度 (可选，但推荐)
        self.proj = nn.Linear(num_heads * head_size, embed_dim)

    def forward(self, x):
        # 拼接所有头的输出
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)

        return out


class FeedForward(nn.Module):
    def __init__(self, embed_dim, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.ReLU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
    

class LayerNorm(nn.Module):
    """
    LayerNorm 的简化实现。注意它与 BatchNorm 的区别：
    BatchNorm 是在 Batch 维度归一化，而 LayerNorm 是在 Channel 维度归一化。
    """
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        # 用nn.Parameter初始化两个可学习参数：gamma (缩放) 和 beta (平移)
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        # 计算 x 在最后一个维度上的均值和方差
        # 执行归一化：(x - mean) / sqrt(var + eps)
        # 应用 gamma 和 beta 进行仿射变换
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)

        x = (x - mean) / torch.sqrt(var + self.eps)
        x = x * self.gamma + self.beta

        return x


class Block(nn.Module):
    """ Transformer Block: 通信 (Attention) + 计算 (FFN) """

    def __init__(self, n_embed, n_head):
        # n_embed: embedding 维度, n_head: 我们想要的头数
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_embed, n_head, head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = LayerNorm(n_embed)
        self.ln2 = LayerNorm(n_embed)

    def forward(self, x):
        # 实现 Pre-Norm 结构下的残差连接
        # 1. x = x + Attention(LayerNorm(x))
        # 2. x = x + FFN(LayerNorm(x))
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, block_size, n_embed):
        super().__init__()

        position = torch.arange(block_size).unsqueeze(1)  # (block_size, 1)

        div_term = torch.exp(
            torch.arange(0, n_embed, 2) * (-torch.log(torch.tensor(10000.0)) / n_embed)
        )  # (n_embed / 2,)

        pe = torch.zeros(block_size, n_embed)

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 不是可学习参数，但会跟随模型保存和移动到 GPU
        self.register_buffer("pe", pe)

    def forward(self, T):
        return self.pe[:T]
    

class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size, n_embed, n_head, n_layer, block_size):
        super().__init__()
        # 1. Token 嵌入表
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        # 2. 位置编码 嵌入表 (让模型知道字符的顺序)
        self.position_encoding = SinusoidalPositionalEncoding(block_size, n_embed)
        # 3. 堆叠多个 Transformer Block
        self.blocks = nn.Sequential(*[Block(n_embed, n_head) for _ in range(n_layer)])
        # 4. 最后的层归一化
        self.ln_f = LayerNorm(n_embed)
        # 5. 语言模型头 (从特征空间映射回词表大小)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # tok_emb: (B, T, C), pos_emb: (T, C)
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_encoding(T)
        x = tok_emb + pos_emb # 融合内容信息和位置信息

        x = self.blocks(x)    # 通过所有 Block
        x = self.ln_f(x)      # 最终归一化
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.reshape(B * T, C)
            targets = targets.reshape(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # 限制上下文长度不能超过 block_size
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] # 只关注最后一个时刻
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
    

# 确认 BPE tokenizer 的词表大小
vocab_size = tokenizer.get_vocab_size()
print(f"vocab_size: {vocab_size}")

# 超参数设置
embed_dim = 512  # embedding dimension
n_head = 8
n_layer = 10  # transformer block 的数量
learning_rate = 3e-4
norm_clip = 1.0
max_iters = 200000

cuda_id = 4
device = torch.device("cuda:{}".format(cuda_id) if torch.cuda.is_available() else "cpu")

gpt_model = GPTLanguageModel(
    vocab_size=vocab_size,
    n_embed=embed_dim,
    n_head=n_head,
    n_layer=n_layer,
    block_size=block_size,
).to(device)

# print(gpt_model)
total_params = sum(p.numel() for p in gpt_model.parameters())
print(f"模型参数数量: {total_params:,}")

opt = torch.optim.AdamW(gpt_model.parameters(), 
                        lr=learning_rate, 
                        weight_decay=1e-2,
                        betas=(0.90, 0.95),
                        eps=1e-8)

warmup_iters = 1000
min_lr = learning_rate * 0.1

warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
    opt,
    start_factor=1e-5,
    end_factor=1.0,
    total_iters=warmup_iters
)

cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    opt,
    T_max=max_iters - warmup_iters,
    eta_min=min_lr
)

scheduler = torch.optim.lr_scheduler.SequentialLR(
    opt,
    schedulers=[warmup_scheduler, cosine_scheduler],
    milestones=[warmup_iters]
)

train_iter = iter(train_loader)
val_iter = iter(val_loader)

# 训练循环
# for i in range(max_iters):
for i in tqdm(range(max_iters)):
    gpt_model.train()

    try:
        batch = next(train_iter)
    except StopIteration:
        train_iter = iter(train_loader)
        batch = next(train_iter)

    xb = batch["input_ids"].to(device)
    yb = batch["labels"].to(device)

    logits, loss = gpt_model(xb, yb)

    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(gpt_model.parameters(), norm_clip)
    opt.step()
    scheduler.step()

    if i % 500 == 0:
        print(f"iter {i}: train loss {loss.item():.4f}")

        val_losses = []
        gpt_model.eval()

        with torch.no_grad():
            try:
                val_batch = next(val_iter)
            except StopIteration:
                val_iter = iter(val_loader)
                val_batch = next(val_iter)
            xb, yb = val_batch["input_ids"].to(device), val_batch["labels"].to(device)
            _, loss = gpt_model(xb, yb)
            val_losses.append(loss.item())
        avg_val_loss = sum(val_losses) / len(val_losses)

        print(f"iter {i}: validation loss {avg_val_loss:.4f}")

print("训练完成，开始评估...")
gpt_model.eval()

val_losses = []
with torch.no_grad():
    for batch in val_loader:
        xb, yb = batch["input_ids"], batch["labels"]
        xb, yb = xb.to(device), yb.to(device)
        _, loss = gpt_model(xb, yb)
        val_losses.append(loss.item())
avg_val_loss = sum(val_losses) / len(val_losses)
print(f"平均验证损失: {avg_val_loss:.4f}")

print("生成示例文本:")
context = "The history of natural language processing (NLP) started in the 1950s"
context_ids = torch.tensor(tokenizer.encode(context).ids, dtype=torch.long, device=device).unsqueeze(0)
generated_ids = gpt_model.generate(context_ids, max_new_tokens=100)[0].tolist()
generated_text = tokenizer.decode(generated_ids)
print(generated_text)


# 评估模型在测试集上的性能
batch_size = 16

test_loader = DataLoader(
    lm_ds["test"],
    batch_size=batch_size,
    shuffle=False
)
test_iter = iter(test_loader)

@torch.no_grad()
def evaluate_metrics(model, data_iter, num_batches=50):
    model.eval()
    total_loss = 0
    total_tokens = 0

    for _ in range(num_batches):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(test_loader)
            batch = next(data_iter)

        xb = batch["input_ids"].to(device)
        yb = batch["labels"].to(device)

        _, loss = model(xb, yb)

        total_loss += loss.item() * batch_size * block_size
        total_tokens += batch_size * block_size

    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)

    model.train()

    return {
        "Loss": avg_loss,
        "Perplexity": perplexity
    }


test_ids = []

for item in lm_ds["test"]:
    test_ids.extend(item["input_ids"])

metrics = evaluate_metrics(gpt_model, test_iter)

print("测试集评估结果:")
print(f"- Cross Entropy Loss: {metrics['Loss']:.4f}")
print(f"- Perplexity: {metrics['Perplexity']:.2f}")


# 保存 checkpoint
save_dir = "model/gpt_wikitext103_layer{}".format(n_layer)
os.makedirs(save_dir, exist_ok=True)

save_path = os.path.join(save_dir, "checkpoint.pt")

total_params = sum(p.numel() for p in gpt_model.parameters())

checkpoint = {
    "model_state_dict": gpt_model.state_dict(),
    "optimizer_state_dict": opt.state_dict(),
    "scheduler_state_dict": scheduler.state_dict(),

    "vocab_size": vocab_size,
    "embed_dim": embed_dim,
    "n_head": n_head,
    "n_layer": n_layer,
    "block_size": block_size,

    "learning_rate": learning_rate,
    "norm_clip": norm_clip,
    "max_iters": max_iters,
    "warmup_iters": warmup_iters,
    "min_lr": min_lr,

    "total_params": total_params,
    "test_loss": metrics["Loss"],
    "test_perplexity": metrics["Perplexity"],
}

torch.save(checkpoint, save_path)

print(f"checkpoint 已保存到: {save_path}")
print(f"模型参数数量: {total_params:,}")
