"""
간단한 임베딩 추출 데모
단일 이미지에서 DETR 임베딩을 추출하는 예제
사용법: cd embedding && python demo_extract_single_image.py --image_path <path> --checkpoint <path>
"""
import sys
sys.path.append('..')

import torch
import torchvision.transforms as T
from PIL import Image
import numpy as np
import argparse
from pathlib import Path

from models import build_model


def extract_embedding_from_image(image_path, model, device):
    """
    단일 이미지에서 임베딩 추출
    
    Args:
        image_path: 이미지 경로
        model: DETR 모델
        device: 실행 디바이스
    
    Returns:
        embeddings: 추출된 임베딩들
    """
    # 이미지 로드 및 전처리
    img = Image.open(image_path).convert('RGB')
    
    transform = T.Compose([
        T.Resize(800),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # Feature 추출을 위한 hook 등록
    features = {}
    
    def backbone_hook(module, input, output):
        if isinstance(output, tuple):
            feat, pos = output
            features['backbone'] = feat[-1].tensors.detach().cpu()
    
    def transformer_hook(module, input, output):
        # Transformer의 출력 저장
        if isinstance(output, tuple) and len(output) >= 2:
            hs, memory = output
            features['decoder'] = hs.detach().cpu()  # [num_layers, B, num_queries, C]
            features['encoder'] = memory.detach().cpu()  # [H*W, B, C]
    
    # Hook 등록
    backbone_handle = model.backbone.register_forward_hook(backbone_hook)
    transformer_handle = model.transformer.register_forward_hook(transformer_hook)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(img_tensor)
    
    # Hook 제거
    backbone_handle.remove()
    transformer_handle.remove()
    
    # 최종 query embeddings 추출
    features['query_embeddings'] = features['decoder'][-1, 0]  # [num_queries, C]
    features['pred_logits'] = outputs['pred_logits'][0].detach().cpu()
    features['pred_boxes'] = outputs['pred_boxes'][0].detach().cpu()
    
    return features


def main():
    parser = argparse.ArgumentParser('DETR Embedding Extraction Demo')
    parser.add_argument('--image_path', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to DETR checkpoint')
    parser.add_argument('--output_path', type=str, default='embedding_output.pt',
                       help='Path to save embeddings')
    
    # Model parameters (DETR 기본값)
    parser.add_argument('--backbone', default='resnet50', type=str)
    parser.add_argument('--dilation', action='store_true')
    parser.add_argument('--position_embedding', default='sine', type=str)
    parser.add_argument('--enc_layers', default=6, type=int)
    parser.add_argument('--dec_layers', default=6, type=int)
    parser.add_argument('--dim_feedforward', default=2048, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--nheads', default=8, type=int)
    parser.add_argument('--num_queries', default=100, type=int)
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--masks', action='store_true')
    parser.add_argument('--aux_loss', dest='aux_loss', action='store_true')
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false')
    parser.set_defaults(aux_loss=True)
    
    # Loss coefficients
    parser.add_argument('--set_cost_class', default=1, type=float)
    parser.add_argument('--set_cost_bbox', default=5, type=float)
    parser.add_argument('--set_cost_giou', default=2, type=float)
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float)
    
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    # 디바이스 설정
    device = torch.device(args.device)
    
    print(f"Loading model from {args.checkpoint}")
    
    # 모델 빌드
    model, _, _ = build_model(args)
    model.to(device)
    
    # 체크포인트 로드
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    print(f"Processing image: {args.image_path}")
    
    # 임베딩 추출
    embeddings = extract_embedding_from_image(args.image_path, model, device)
    
    # 결과 출력
    print("\nExtracted embeddings:")
    for key, value in embeddings.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: shape = {value.shape}, dtype = {value.dtype}")
    
    # 저장
    torch.save(embeddings, args.output_path)
    print(f"\nSaved embeddings to {args.output_path}")
    
    # Query embeddings 통계
    query_emb = embeddings['query_embeddings']
    print(f"\nQuery Embeddings Statistics:")
    print(f"  Shape: {query_emb.shape}")
    print(f"  Mean: {query_emb.mean():.4f}")
    print(f"  Std: {query_emb.std():.4f}")
    print(f"  Min: {query_emb.min():.4f}")
    print(f"  Max: {query_emb.max():.4f}")
    
    # Detection 결과
    pred_logits = embeddings['pred_logits']
    pred_boxes = embeddings['pred_boxes']
    
    # Person class (id=0 for COCO person)
    probs = pred_logits.softmax(-1)[:, 0]  # person class probability
    keep = probs > 0.5
    
    print(f"\nDetection Results:")
    print(f"  Total queries: {len(probs)}")
    print(f"  High confidence detections (>0.5): {keep.sum().item()}")
    
    if keep.sum() > 0:
        print(f"\n  Top 5 detections:")
        top_indices = probs.topk(min(5, len(probs)))[1]
        for i, idx in enumerate(top_indices):
            print(f"    {i+1}. Query {idx}: confidence = {probs[idx]:.4f}, "
                  f"box = {pred_boxes[idx].tolist()}")


if __name__ == '__main__':
    main()
