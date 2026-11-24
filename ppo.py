# Proximal Policy Optimization

import copy
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from rm import RewardModel          

POLICY = "sft_ckpt"                 
RM_CKPT = "rm.pt"
STEPS, ROLLOUT_BS, MINI_BS, PPO_EPOCHS = 50, 16, 4, 4
MAX_PROMPT, MAX_NEW = 128, 64
LR, VF_LR = 5e-6, 1e-4
KL_COEF, CLIP, VF_COEF, GAMMA, LAM = 0.05, 0.2, 0.1, 1.0, 0.95
device = "cuda" if torch.cuda.is_available() else "cpu"


tok = AutoTokenizer.from_pretrained(POLICY)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"           

policy = AutoModelForCausalLM.from_pretrained(POLICY).to(device)
ref = copy.deepcopy(policy).eval()
for p in ref.parameters():
    p.requires_grad = False

value_head = nn.Linear(policy.config.hidden_size, 1).to(device)
nn.init.zeros_(value_head.weight); nn.init.zeros_(value_head.bias)

rm = RewardModel(POLICY).to(device)
rm.load_state_dict(torch.load(RM_CKPT, map_location=device))
rm.eval()
for p in rm.parameters():
    p.requires_grad = False

opt = torch.optim.AdamW([
    {"params": policy.parameters(), "lr": LR},
    {"params": value_head.parameters(), "lr": VF_LR},
])

ds = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split="train_prefs[:2000]")
prompts = [ex["prompt"] + "\n\nAnswer:" for ex in ds]
prompts = [p for p in prompts if len(tok(p).input_ids) <= MAX_PROMPT]
print(f"{len(prompts)} prompts")



def logprobs_and_values(model, seq, attn, P, vhead=None):
    """Log-prob (and value) of each response token. seq = [left-padded prompt | response]."""
    pos = (attn.cumsum(-1) - 1).clamp(min=0)                      
    out = model(input_ids=seq, attention_mask=attn, position_ids=pos,
                output_hidden_states=vhead is not None)
    logits = out.logits[:, P - 1:-1].float()                      
    lp = torch.log_softmax(logits, -1).gather(-1, seq[:, P:].unsqueeze(-1)).squeeze(-1)
    values = vhead(out.hidden_states[-1][:, P - 1:-1]).squeeze(-1) if vhead is not None else None
    return lp, values


def reward_scores(enc, resp, mask):
    """RM expects right-padded [prompt | response]; rebuild that from the left-padded pieces."""
    seqs = [torch.cat([enc.input_ids[i][enc.attention_mask[i].bool()], resp[i][mask[i].bool()]])
            for i in range(resp.size(0))]
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), tok.pad_token_id, device=device)
    am = torch.zeros_like(ids)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = s; am[i, :len(s)] = 1
    return rm(ids, am)


@torch.no_grad()
def rollout(prompt_texts):
    enc = tok(prompt_texts, return_tensors="pt", padding=True).to(device)
    P = enc.input_ids.size(1)
    seq = policy.generate(**enc, max_new_tokens=MAX_NEW, do_sample=True, top_k=0, top_p=1.0,
                          temperature=1.0, pad_token_id=tok.pad_token_id)
    resp = seq[:, P:]
    is_eos = resp == tok.eos_token_id
    mask = ((is_eos.cumsum(-1) - is_eos.long()) == 0).float()     
    attn = torch.cat([enc.attention_mask, mask.long()], dim=1)

    old_lp, values = logprobs_and_values(policy, seq, attn, P, value_head)
    ref_lp, _ = logprobs_and_values(ref, seq, attn, P)
    score = reward_scores(enc, resp, mask)

    kl = old_lp - ref_lp
    rewards = -KL_COEF * kl * mask                              
    last = mask.sum(1).long() - 1
    rewards[torch.arange(resp.size(0)), last] += score            
    return dict(seq=seq, attn=attn, P=P, mask=mask, old_lp=old_lp, values=values,
                rewards=rewards, score=score, kl=(kl * mask).sum(1) / mask.sum(1), resp=resp)


def compute_gae(values, rewards, mask):
    B, T = rewards.shape
    adv = torch.zeros_like(rewards)
    running = torch.zeros(B, device=rewards.device)
    for t in reversed(range(T)):
        next_m = mask[:, t + 1] if t + 1 < T else torch.zeros(B, device=rewards.device)
        next_v = values[:, t + 1] * next_m if t + 1 < T else torch.zeros(B, device=rewards.device)
        delta = rewards[:, t] + GAMMA * next_v - values[:, t]
        running = delta + GAMMA * LAM * next_m * running
        adv[:, t] = running
    returns = adv + values
    mean = (adv * mask).sum() / mask.sum()
    std = (((adv - mean) ** 2 * mask).sum() / mask.sum()).sqrt()
    return ((adv - mean) / (std + 1e-8)) * mask, returns



for step in range(STEPS):
    idx = torch.randperm(len(prompts))[:ROLLOUT_BS].tolist()
    batch = rollout([prompts[i] for i in idx])
    adv, ret = compute_gae(batch["values"], batch["rewards"], batch["mask"])

    for _ in range(PPO_EPOCHS):
        perm = torch.randperm(ROLLOUT_BS, device=device)
        for i in range(0, ROLLOUT_BS, MINI_BS):
            b = perm[i:i + MINI_BS]
            lp, v = logprobs_and_values(policy, batch["seq"][b], batch["attn"][b], batch["P"], value_head)
            m = batch["mask"][b]
            ratio = torch.exp(lp - batch["old_lp"][b])
            pg = torch.max(-adv[b] * ratio, -adv[b] * ratio.clamp(1 - CLIP, 1 + CLIP))
            pg_loss = (pg * m).sum() / m.sum()
            vf_loss = (((v - ret[b]) ** 2) * m).sum() / m.sum()
            loss = pg_loss + VF_COEF * vf_loss
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(list(policy.parameters()) + list(value_head.parameters()), 1.0)
            opt.step()

    print(f"step {step} reward {batch['score'].mean():.3f} kl {batch['kl'].mean():.3f} "
          f"len {batch['mask'].sum(1).mean():.1f} pg {pg_loss.item():.3f} vf {vf_loss.item():.3f}")
    if step % 10 == 0:
        r0 = batch["resp"][0][batch["mask"][0].bool()]
        print("  sample:", tok.decode(r0, skip_special_tokens=True)[:200].replace("\n", " "))

policy.save_pretrained("ppo_ckpt"); tok.save_pretrained("ppo_ckpt")