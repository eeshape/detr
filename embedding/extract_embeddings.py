# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
FACET 데이터셋에서 DETR 임베딩 벡터 f를 추출하는 스크립트
"""
import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

import sys
sys.path.append('..')

import datasets
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from models import build_model


def get_args_parser():
    parser = argparse.ArgumentParser('DETR embedding extraction', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=100, type=int,
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true',
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str, required=True)
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')
    parser.add_argument('--skin', type=str, default=None,
                        help='Skin annotation file name (e.g., skin1, skin2, ..., skin10)')
    parser.add_argument('--gender', type=str, default=None,
                        help='Gender annotation file name (e.g., men, women)')
    parser.add_argument('--light', type=str, default=None, choices=['well', 'dimly'],
                        help='Lighting condition (well for well_lit, dimly for dimly_lit)')

    # Embedding extraction parameters
    parser.add_argument('--extract_type', default='all', type=str,
                        choices=['backbone', 'encoder', 'decoder', 'all'],
                        help='Type of embeddings to extract')
    parser.add_argument('--output_dir', default='./embeddings_output',
                        help='path where to save embeddings')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


@torch.no_grad()
def extract_embeddings(model, data_loader, device, output_dir, extract_type='all'):
    """
    데이터셋에서 임베딩 추출
    
    Args:
        model: DETR 모델
        data_loader: 데이터 로더
        device: 실행 디바이스
        output_dir: 저장 경로
        extract_type: 추출할 임베딩 타입
    """
    model.eval()
    
    # 저장 디렉토리 생성
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Hook을 통해 중간 feature 추출
    features_dict = {}
    
    def get_backbone_hook(module, input, output):
        if isinstance(output, tuple):
            feat, pos = output
            features_dict['backbone_features'] = feat[-1].tensors.detach().cpu()
    
    def get_transformer_hook(module, input, output):
        if isinstance(output, tuple) and len(output) >= 2:
            hs, memory = output
            features_dict['decoder_output'] = hs.detach().cpu()
            features_dict['encoder_output'] = memory.detach().cpu()
    
    # Hook 등록
    handles = []
    if extract_type in ['backbone', 'all']:
        handles.append(model.backbone.register_forward_hook(get_backbone_hook))
    if extract_type in ['encoder', 'decoder', 'all']:
        handles.append(model.transformer.register_forward_hook(get_transformer_hook))
    
    # 임베딩 저장용
    all_embeddings = {
        'image_ids': [],
        'embeddings': []
    }
    
    print(f"Extracting embeddings (type: {extract_type}) to {output_dir}")
    
    for samples, targets in tqdm(data_loader, desc="Extracting embeddings"):
        samples = samples.to(device)
        
        # Forward pass
        features_dict = {}
        outputs = model(samples)
        
        # 배치 내 각 이미지에 대해 저장
        batch_size = len(targets)
        for i in range(batch_size):
            image_id = targets[i]['image_id'].item()
            
            # 각 이미지별 임베딩
            image_embeddings = {}
            
            # Backbone features
            if 'backbone_features' in features_dict:
                image_embeddings['backbone_features'] = features_dict['backbone_features'][i]
            
            # Encoder output
            if 'encoder_output' in features_dict:
                # encoder output shape: [H*W, B, C] -> [H*W, C] for image i
                image_embeddings['encoder_output'] = features_dict['encoder_output'][:, i, :]
            
            # Decoder output
            if 'decoder_output' in features_dict:
                # decoder output shape: [num_layers, B, num_queries, C]
                image_embeddings['decoder_output'] = features_dict['decoder_output'][:, i, :, :]
                # Query embeddings (마지막 레이어만)
                image_embeddings['query_embeddings'] = features_dict['decoder_output'][-1, i, :, :]
            
            # Predictions
            image_embeddings['pred_logits'] = outputs['pred_logits'][i].detach().cpu()
            image_embeddings['pred_boxes'] = outputs['pred_boxes'][i].detach().cpu()
            
            # 이미지 ID를 파일명으로 사용 (앞의 0 제거)
            save_path = output_path / f'{image_id}.pt'
            torch.save(image_embeddings, save_path)
            
            all_embeddings['image_ids'].append(image_id)
    
    # Hook 제거
    for handle in handles:
        handle.remove()
    
    # 메타데이터 저장
    # 샘플 임베딩 로드하여 키 목록 확인
    if all_embeddings['image_ids']:
        sample_id = all_embeddings['image_ids'][0]
        sample_emb = torch.load(output_path / f'{sample_id}.pt')
        embedding_keys = list(sample_emb.keys())
    else:
        embedding_keys = []
    
    metadata = {
        'image_ids': all_embeddings['image_ids'],
        'num_images': len(all_embeddings['image_ids']),
        'extract_type': extract_type,
        'embedding_keys': embedding_keys
    }
    
    with open(output_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nSaved {len(all_embeddings['image_ids'])} embeddings to {output_dir}")
    print(f"Embedding keys: {metadata['embedding_keys']}")
    
    return metadata


def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))
    
    print(args)
    
    device = torch.device(args.device)
    
    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # 모델 빌드
    model, criterion, postprocessors = build_model(args)
    model.to(device)
    
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    
    # 데이터셋 빌드
    dataset_val = build_dataset(image_set='val', args=args)
    
    if args.distributed:
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    
    data_loader_val = DataLoader(
        dataset_val, args.batch_size, sampler=sampler_val,
        drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers
    )
    
    # 체크포인트 로드
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        print(f"Loaded checkpoint from {args.resume}")
    else:
        print("Warning: No checkpoint provided. Using randomly initialized weights.")
    
    # 임베딩 추출
    print("Start extracting embeddings")
    start_time = time.time()
    
    metadata = extract_embeddings(
        model_without_ddp, data_loader_val, device, 
        args.output_dir, args.extract_type
    )
    
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Extraction time {}'.format(total_time_str))
    
    return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR embedding extraction script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
