"""
추출된 임베딩 벡터 f를 로드하고 분석하는 유틸리티
사용법: cd embedding && python analyze_embeddings.py
"""
import torch
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt

try:
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available. Install with: pip install scikit-learn")


class EmbeddingAnalyzer:
    """추출된 임베딩을 분석하는 클래스"""
    
    def __init__(self, embedding_dir):
        """
        Args:
            embedding_dir: 임베딩이 저장된 디렉토리 경로
        """
        self.embedding_dir = Path(embedding_dir)
        self.metadata = self._load_metadata()
        
    def _load_metadata(self):
        """메타데이터 로드"""
        metadata_path = self.embedding_dir / 'metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                return json.load(f)
        return None
    
    def load_embedding(self, image_id):
        """특정 이미지의 임베딩 로드"""
        embedding_path = self.embedding_dir / f'embedding_{image_id:012d}.pt'
        if embedding_path.exists():
            return torch.load(embedding_path)
        return None
    
    def load_all_embeddings(self, embedding_key='query_embeddings'):
        """
        모든 임베딩을 로드
        
        Args:
            embedding_key: 로드할 임베딩 타입
                - 'backbone_features': backbone feature map
                - 'encoder_output': encoder output
                - 'decoder_output': decoder hidden states (모든 레이어)
                - 'query_embeddings': 최종 query embeddings (가장 유용)
        
        Returns:
            embeddings: 임베딩 리스트
            image_ids: 이미지 ID 리스트
        """
        if not self.metadata:
            raise ValueError("Metadata not found")
        
        embeddings = []
        image_ids = []
        
        for image_id in self.metadata['image_ids']:
            emb = self.load_embedding(image_id)
            if emb and embedding_key in emb:
                embeddings.append(emb[embedding_key])
                image_ids.append(image_id)
        
        return embeddings, image_ids
    
    def get_query_embeddings_as_matrix(self):
        """
        Query embeddings를 행렬 형태로 반환
        
        Returns:
            matrix: [num_images * num_queries, embedding_dim] 형태의 행렬
            image_ids: 각 query가 속한 이미지 ID
            query_indices: 각 query의 인덱스
        """
        embeddings, img_ids = self.load_all_embeddings('query_embeddings')
        
        all_queries = []
        all_image_ids = []
        all_query_indices = []
        
        for img_id, emb in zip(img_ids, embeddings):
            # emb shape: [num_queries, embedding_dim]
            num_queries = emb.shape[0]
            for q_idx in range(num_queries):
                all_queries.append(emb[q_idx].numpy())
                all_image_ids.append(img_id)
                all_query_indices.append(q_idx)
        
        matrix = np.stack(all_queries, axis=0)
        return matrix, all_image_ids, all_query_indices
    
    def visualize_embeddings_tsne(self, embedding_key='query_embeddings', 
                                   n_components=2, save_path=None):
        """
        t-SNE를 사용하여 임베딩 시각화
        
        Args:
            embedding_key: 시각화할 임베딩 타입
            n_components: 차원 축소 후 차원 수 (2 또는 3)
            save_path: 저장 경로 (None이면 표시만)
        """
        if not SKLEARN_AVAILABLE:
            print("Error: sklearn is required for t-SNE. Install with: pip install scikit-learn")
            return None
            
        if embedding_key == 'query_embeddings':
            # Query embeddings는 특별 처리
            matrix, image_ids, query_indices = self.get_query_embeddings_as_matrix()
        else:
            embeddings, image_ids = self.load_all_embeddings(embedding_key)
            # Flatten if needed
            matrix = []
            for emb in embeddings:
                if len(emb.shape) > 2:
                    # 평균 pooling
                    emb = emb.mean(dim=tuple(range(len(emb.shape)-1)))
                matrix.append(emb.numpy())
            matrix = np.stack(matrix, axis=0)
        
        print(f"Applying t-SNE to {matrix.shape[0]} embeddings of dimension {matrix.shape[1]}")
        
        # t-SNE 적용
        tsne = TSNE(n_components=n_components, random_state=42, perplexity=30)
        embeddings_2d = tsne.fit_transform(matrix)
        
        # 시각화
        plt.figure(figsize=(10, 8))
        if n_components == 2:
            plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.5, s=10)
            plt.xlabel('t-SNE 1')
            plt.ylabel('t-SNE 2')
        else:
            from mpl_toolkits.mplot3d import Axes3D
            ax = plt.gcf().add_subplot(111, projection='3d')
            ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], embeddings_2d[:, 2], 
                      alpha=0.5, s=10)
            ax.set_xlabel('t-SNE 1')
            ax.set_ylabel('t-SNE 2')
            ax.set_zlabel('t-SNE 3')
        
        plt.title(f't-SNE visualization of {embedding_key}')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
        else:
            plt.show()
        
        return embeddings_2d
    
    def visualize_embeddings_pca(self, embedding_key='query_embeddings', 
                                  n_components=2, save_path=None):
        """
        PCA를 사용하여 임베딩 시각화
        
        Args:
            embedding_key: 시각화할 임베딩 타입
            n_components: 차원 축소 후 차원 수 (2 또는 3)
            save_path: 저장 경로
        """
        if not SKLEARN_AVAILABLE:
            print("Error: sklearn is required for PCA. Install with: pip install scikit-learn")
            return None, None
            
        if embedding_key == 'query_embeddings':
            matrix, image_ids, query_indices = self.get_query_embeddings_as_matrix()
        else:
            embeddings, image_ids = self.load_all_embeddings(embedding_key)
            matrix = []
            for emb in embeddings:
                if len(emb.shape) > 2:
                    emb = emb.mean(dim=tuple(range(len(emb.shape)-1)))
                matrix.append(emb.numpy())
            matrix = np.stack(matrix, axis=0)
        
        print(f"Applying PCA to {matrix.shape[0]} embeddings of dimension {matrix.shape[1]}")
        
        # PCA 적용
        pca = PCA(n_components=n_components)
        embeddings_reduced = pca.fit_transform(matrix)
        
        print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
        
        # 시각화
        plt.figure(figsize=(10, 8))
        if n_components == 2:
            plt.scatter(embeddings_reduced[:, 0], embeddings_reduced[:, 1], alpha=0.5, s=10)
            plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
            plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
        else:
            from mpl_toolkits.mplot3d import Axes3D
            ax = plt.gcf().add_subplot(111, projection='3d')
            ax.scatter(embeddings_reduced[:, 0], embeddings_reduced[:, 1], 
                      embeddings_reduced[:, 2], alpha=0.5, s=10)
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
            ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.2%})')
        
        plt.title(f'PCA visualization of {embedding_key}')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
        else:
            plt.show()
        
        return embeddings_reduced, pca
    
    def get_embedding_statistics(self, embedding_key='query_embeddings'):
        """임베딩 통계 정보 반환"""
        if embedding_key == 'query_embeddings':
            matrix, _, _ = self.get_query_embeddings_as_matrix()
        else:
            embeddings, _ = self.load_all_embeddings(embedding_key)
            matrix = []
            for emb in embeddings:
                if len(emb.shape) > 2:
                    emb = emb.mean(dim=tuple(range(len(emb.shape)-1)))
                matrix.append(emb.numpy())
            matrix = np.stack(matrix, axis=0)
        
        stats = {
            'shape': matrix.shape,
            'mean': np.mean(matrix, axis=0),
            'std': np.std(matrix, axis=0),
            'min': np.min(matrix, axis=0),
            'max': np.max(matrix, axis=0),
            'norm_mean': np.mean(np.linalg.norm(matrix, axis=1)),
            'norm_std': np.std(np.linalg.norm(matrix, axis=1)),
        }
        
        return stats


def example_usage():
    """사용 예제"""
    # 임베딩 분석기 초기화
    analyzer = EmbeddingAnalyzer('./embeddings_output')
    
    # 메타데이터 출력
    print("Metadata:", analyzer.metadata)
    
    if not analyzer.metadata or not analyzer.metadata.get('image_ids'):
        print("No embeddings found. Please run extract_embeddings.py first.")
        return
    
    # 통계 정보
    stats = analyzer.get_embedding_statistics('query_embeddings')
    print("\nEmbedding Statistics:")
    print(f"Shape: {stats['shape']}")
    print(f"Mean norm: {stats['norm_mean']:.4f} ± {stats['norm_std']:.4f}")
    
    if SKLEARN_AVAILABLE:
        # PCA 시각화
        print("\nGenerating PCA visualization...")
        analyzer.visualize_embeddings_pca(
            'query_embeddings', 
            n_components=2, 
            save_path='./pca_visualization.png'
        )
        
        # t-SNE 시각화
        print("\nGenerating t-SNE visualization...")
        analyzer.visualize_embeddings_tsne(
            'query_embeddings', 
            n_components=2, 
            save_path='./tsne_visualization.png'
        )
    else:
        print("\nSkipping visualizations (sklearn not available)")
    
    # 특정 이미지의 임베딩 로드
    image_id = analyzer.metadata['image_ids'][0]
    embedding = analyzer.load_embedding(image_id)
    if embedding:
        print(f"\nEmbedding keys for image {image_id}:", embedding.keys())
        if 'query_embeddings' in embedding:
            print(f"Query embeddings shape: {embedding['query_embeddings'].shape}")


if __name__ == '__main__':
    example_usage()
