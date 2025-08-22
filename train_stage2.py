import typing as tp
from itertools import chain
import argparse
import random
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.distributed as dist
from tqdm import tqdm
from transformers import BertTokenizer, BertModel
import warnings
import logging

warnings.simplefilter(action='ignore', category=FutureWarning)

from data.stage2 import SimpleDataset, collate_fn
from models.audio_model import AudioModel
from models.mlp import MLP

MEL_SPEC_MAX_LEN = 1251
MEL_SPEC_DIM = 128
HIDDEN_DIM = 768
OUTPUT_DIM = 100


def ce_loss(logits: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    loss1 = ce_loss(similarity)
    loss2 = ce_loss(similarity.t())
    return 0.5 * loss1 + 0.5 * loss2


def contrastive_loss_distributed(
        emb1: torch.Tensor,
        emb2: torch.Tensor,
        logit_scale: torch.Tensor
):
    with torch.no_grad():
        all_emb1 = [torch.zeros_like(emb1) for _ in range(dist.get_world_size())]
        all_emb2 = [torch.zeros_like(emb2) for _ in range(dist.get_world_size())]
        torch.distributed.all_gather(all_emb1, emb1)
        torch.distributed.all_gather(all_emb2, emb2)

    all_emb1[dist.get_rank()] = emb1
    all_emb2[dist.get_rank()] = emb2

    tensor_out1 = torch.cat(all_emb1, dim=0)   # [B * world_size, D]
    tensor_out2 = torch.cat(all_emb2, dim=0)   # [B * world_size, D]

    emb1_normed = tensor_out1 / tensor_out1.norm(p=2, dim=-1, keepdim=True)
    emb2_normed = tensor_out2 / tensor_out2.norm(p=2, dim=-1, keepdim=True)

    cos_sim = torch.matmul(emb1_normed, emb2_normed.t())
    logits = cos_sim * logit_scale.exp()
    loss = clip_loss(logits)

    return loss


class ModelWrapper(torch.nn.Module):
    def __init__(
            self,
            audio_model,
            text_model,
            audio_mlp,  # dimensionality reduction
            text_mlp,  # dimensionality reduction
            fusion_mlp,  # fusion layer
            logit_scale_init_value: float,
            loss_weights: tp.List[float],
    ):
        super().__init__()
        self.audio_model = audio_model
        self.text_model = text_model
        self.audio_mlp = audio_mlp
        self.text_mlp = text_mlp
        self.fusion_mlp = fusion_mlp

        # audio emb <--> text emb
        self.tau_1 = nn.Parameter(torch.ones([]) * logit_scale_init_value)

        # trigger song audio emb <--> recalled song audio emb
        self.tau_2 = nn.Parameter(torch.ones([]) * logit_scale_init_value)

        # trigger song audio emb <--> recalled song fusion emb or
        # trigger song fusion emb <--> recalled song audio emb
        self.tau_3 = nn.Parameter(torch.ones([]) * logit_scale_init_value)

        self.loss_weights = loss_weights

    def forward(
            self,
            mel_spec_1: torch.FloatTensor,
            mel_spec_2: torch.FloatTensor,
            text_inputs_1: tp.Dict[str, torch.LongTensor],
            text_inputs_2: tp.Dict[str, torch.LongTensor],
    ):
        # ======================== audio tower ================================
        audio_emb_1 = self.audio_model(mel_spec_1)  # [B, L, D]
        audio_emb_1 = torch.mean(audio_emb_1, dim=1)  # [B, D]
        audio_emb_1 = self.audio_mlp(audio_emb_1)  # [B, d]

        audio_emb_2 = self.audio_model(mel_spec_2)  # [B, L, D]
        audio_emb_2 = torch.mean(audio_emb_2, dim=1)  # [B, D]
        audio_emb_2 = self.audio_mlp(audio_emb_2)  # [B, d]

        # ======================== text tower =================================
        text_emb_1 = self.text_model(
            input_ids=text_inputs_1['input_ids'],
            token_type_ids=text_inputs_1['token_type_ids'],
            attention_mask=text_inputs_1['attention_mask']
        ).pooler_output
        text_emb_1 = self.text_mlp(text_emb_1)  # [B, d]

        text_emb_2 = self.text_model(
            input_ids=text_inputs_2['input_ids'],
            token_type_ids=text_inputs_2['token_type_ids'],
            attention_mask=text_inputs_2['attention_mask']
        ).pooler_output  # [B, D]
        text_emb_2 = self.text_mlp(text_emb_2)  # [B, d]

        # ======================== fusion =====================================
        fusion_emb_1 = torch.cat([audio_emb_1, text_emb_1], dim=1)
        fusion_emb_1 = self.fusion_mlp(fusion_emb_1)

        fusion_emb_2 = torch.cat([audio_emb_2, text_emb_2], dim=1)
        fusion_emb_2 = self.fusion_mlp(fusion_emb_2)

        # ======================= loss ========================================
        # align loss between audio emb and text emb
        audio_emb = torch.cat([audio_emb_1, audio_emb_2], dim=0)  # [2B, d]
        text_emb = torch.cat([text_emb_1, text_emb_2], dim=0)  # [2B, d]
        loss_1 = contrastive_loss_distributed(audio_emb, text_emb, self.tau_1)

        # align loss between audio embs
        loss_2 = contrastive_loss_distributed(audio_emb_1, audio_emb_2, self.tau_2)

        # align loss between audio emb and fusion emb
        loss_3 = contrastive_loss_distributed(audio_emb_1, fusion_emb_2, self.tau_3)
        loss_3_ = contrastive_loss_distributed(audio_emb_2, fusion_emb_1, self.tau_3)
        loss_3 = (loss_3 + loss_3_) / 2

        loss = loss_1 * self.loss_weights[0] + loss_2 * self.loss_weights[1] + loss_3 * self.loss_weights[2]

        return loss


def train(args, model, dataloader, optimizer, **kargs):
    model.train()

    logger = kargs['logger']
    epoch = kargs['epoch']
    device = kargs['device']

    for i, (mel_spec1, mel_spec2, encoded_text_inputs1, encoded_text_inputs2) in enumerate(dataloader, 1):
        mel_spec1, mel_spec2, encoded_text_inputs1, encoded_text_inputs2 = (
            mel_spec1.to(device), mel_spec2.to(device),
            encoded_text_inputs1.to(device), encoded_text_inputs2.to(device)
        )

        loss = model(mel_spec1, mel_spec2, encoded_text_inputs1, encoded_text_inputs2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if args.local_rank == 0 and i % args.log_interval == 0 and logger is not None:
            logger.info(f'Train stage. epoch {epoch}, batch {i}/{len(dataloader)}, loss: {loss.item():.4f}')


@torch.no_grad()
def evaluate(args, model, dataloader, **kargs):
    model.train()

    logger = kargs['logger']
    epoch = kargs['epoch']
    device = kargs['device']

    # accumulated metrics
    loss_acc = 0.

    cnt = 0
    for i, (mel_spec1, mel_spec2, encoded_text_inputs1, encoded_text_inputs2) in tqdm(enumerate(dataloader, 1)):
        mel_spec1, mel_spec2, encoded_text_inputs1, encoded_text_inputs2 = (
            mel_spec1.to(device), mel_spec2.to(device),
            encoded_text_inputs1.to(device), encoded_text_inputs2.to(device)
        )

        loss = model(mel_spec1, mel_spec2, encoded_text_inputs1, encoded_text_inputs2)

        loss_acc += loss.item()
        cnt += 1

    loss = loss_acc / cnt

    if args.local_rank == 0 and logger is not None:
        logger.info(f'Evaluate stage. epoch {epoch}, loss: {loss:.4f}')


def main():
    # -------------------------------- hyper-parameters -------------------------------------------
    parser = argparse.ArgumentParser(description='PyTorch Training Setting')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--workers-count', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--base-lr', type=float, default=1e-5)
    parser.add_argument('--log-interval', type=int, default=1, help='i.e. log_every_n_steps')
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--save-model-dir', type=str, default='ckpts', help='directory of saving model')
    parser.add_argument('--log-dir', type=str, default='logs', help='directory of logs')
    parser.add_argument('--logit-scale-init-value', type=float, default=2.6)
    parser.add_argument('--local-rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument('--train-data-ratio', type=float, default=0.9)
    parser.add_argument('--save-period', type=int, default=1)
    args = parser.parse_args()

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    if not os.path.exists(args.save_model_dir):
        os.makedirs(args.save_model_dir)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)
    random.seed(args.seed)

    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)

    logger = None
    if args.local_rank == 0:
        logger = logging.getLogger(__name__)

    # ==== models ====
    # audio model
    audio_model = AudioModel(input_dim=MEL_SPEC_DIM)
    audio_model.load_state_dict(torch.load('audio-model-stage-1.pth', map_location=device))

    # text model
    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-uncased")
    text_model = BertModel.from_pretrained("bert-base-multilingual-uncased")
    text_model.load_state_dict(torch.load('text-model-stage-1.pth', map_location=device))

    # MLPs
    audio_mlp = MLP([HIDDEN_DIM, HIDDEN_DIM // 2, HIDDEN_DIM // 4, OUTPUT_DIM])
    text_mlp = MLP([HIDDEN_DIM, HIDDEN_DIM // 2, HIDDEN_DIM // 4, OUTPUT_DIM])
    fusion_mlp = MLP([OUTPUT_DIM * 2, OUTPUT_DIM * 2, OUTPUT_DIM, OUTPUT_DIM])

    model = ModelWrapper(
        audio_model,
        text_model,
        audio_mlp,
        text_mlp,
        fusion_mlp,
        args.logit_scale_init_value,
        loss_weights=[1., 1., 1.],
    ).cuda()

    model = torch.nn.parallel.DistributedDataParallel(model)

    # ==== datasets ====
    file = "similar-song-pair-dataset.parquet"
    dataset = SimpleDataset(dataset_file=file, inp_feat_max_len=MEL_SPEC_MAX_LEN, padding=True)

    num_train_samples = int(len(dataset) * args.train_data_ratio)
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [num_train_samples, len(dataset) - num_train_samples]
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers_count,
        drop_last=True,
        sampler=train_sampler,
        collate_fn=lambda batch: collate_fn(batch, bert_tokenizer),
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers_count,
        drop_last=True,
        sampler=test_sampler,
        collate_fn=lambda batch: collate_fn(batch, bert_tokenizer),
    )

    if logger is not None:
        logger.info(f'num_all_samples: {len(dataset)}')
        logger.info(f'num_train_samples: {len(train_dataset)}')
        logger.info(f'num_test_samples: {len(test_dataset)}')

    # ==== optimizer ====
    optimizer = torch.optim.Adam([
        {
            'params': chain(
                model.module.audio_model.parameters(),
                model.module.text_model.parameters()
            )
        },
        {
            'params': chain(
                model.module.audio_mlp.parameters(),
                model.module.text_mlp.parameters(),
                model.module.fusion_mlp.parameters(),
                [model.module.tau_1, model.module.tau_2, model.module.tau_3],
            ),
            'lr': args.lr
        }
    ], args.base_lr)

    if logger is not None:
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f'num_params: {num_params}')

    for epoch in range(1, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        train(args, model, train_dataloader, optimizer, logger=logger, epoch=epoch, device=device)
        evaluate(args, model, test_dataloader, logger=logger, epoch=epoch, device=device)

        # save checkpoint
        if args.local_rank == 0 and epoch % args.save_period == 0:
            torch.save(model.module.state_dict(), os.path.join(args.save_model_dir, f'model-{epoch}.pth'))


if __name__ == '__main__':
    main()
