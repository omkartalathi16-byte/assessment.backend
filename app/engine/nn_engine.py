# app/engine/nn_engine.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import logging
from dataclasses import dataclass
from enum import Enum
import pickle
from pathlib import Path
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class NNConfig:
    """Configuration for neural network."""
    input_dim: int = 300  # Embedding dimension
    hidden_dims: List[int] = None  # Default: [256, 128, 64]
    dropout: float = 0.2
    activation: str = "relu"
    batch_norm: bool = True
    num_heads: int = 8  # For attention
    learning_rate: float = 1e-3
    batch_size: int = 32
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 128, 64]


class PositionalEncoding(nn.Module):
    """Positional encoding for sequences."""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0), :]


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask=None):
        batch_size = query.shape[0]
        
        # Linear projections
        Q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)
        
        # Apply attention to values
        out = torch.matmul(attention, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """Transformer encoder block."""
    
    def __init__(self, embed_dim: int, num_heads: int, ff_dim: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask=None):
        # Self-attention
        attn_out = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        
        return x


class SemanticNN(nn.Module):
    """Neural network for semantic analysis."""
    
    def __init__(self, config: NNConfig):
        super().__init__()
        self.config = config
        
        # Input projection
        self.input_proj = nn.Linear(config.input_dim, config.hidden_dims[0])
        
        # Transformer encoder
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=config.hidden_dims[0],
                num_heads=config.num_heads,
                ff_dim=config.hidden_dims[0] * 4,
                dropout=config.dropout
            )
            for _ in range(2)
        ])
        
        # Dense layers
        layers = []
        current_dim = config.hidden_dims[0]
        
        for hidden_dim in config.hidden_dims[1:]:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim) if config.batch_norm else nn.Identity(),
                self._get_activation(config.activation),
                nn.Dropout(config.dropout)
            ])
            current_dim = hidden_dim
        
        self.hidden_layers = nn.Sequential(*layers)
        
        # Output heads
        self.similarity_head = nn.Sequential(
            nn.Linear(current_dim * 2, current_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(current_dim, 1),
            nn.Sigmoid()
        )
        
        self.concept_scorer = nn.Sequential(
            nn.Linear(current_dim, current_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(current_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(config.hidden_dims[0])
        
        # Initialize weights
        self._init_weights()
    
    def _get_activation(self, activation: str):
        """Get activation function."""
        activations = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(0.1),
            "gelu": nn.GELU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid()
        }
        return activations.get(activation, nn.ReLU())
    
    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def encode(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode input to embedding."""
        # Project input
        x = self.input_proj(x)
        
        # Add positional encoding if sequence
        if x.dim() == 3:  # [batch, seq_len, features]
            x = self.pos_encoding(x)
            
            # Apply transformer blocks
            for block in self.transformer_blocks:
                x = block(x, mask)
            
            # Use CLS token or mean pooling
            x = x.mean(dim=1)  # Mean pooling
        else:
            # For single vectors, add sequence dimension
            x = x.unsqueeze(1)
            x = self.pos_encoding(x)
            
            for block in self.transformer_blocks:
                x = block(x, mask)
            
            x = x.squeeze(1)
        
        # Apply hidden layers
        x = self.hidden_layers(x)
        
        return x
    
    def forward(
        self, 
        text1: torch.Tensor, 
        text2: torch.Tensor,
        mask1: Optional[torch.Tensor] = None,
        mask2: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            text1: First text embeddings [batch, seq_len, features] or [batch, features]
            text2: Second text embeddings
            mask1: Optional mask for text1
            mask2: Optional mask for text2
        
        Returns:
            similarity: Similarity score [batch, 1]
            embedding1: Encoded text1 [batch, features]
            embedding2: Encoded text2 [batch, features]
        """
        # Encode both texts
        emb1 = self.encode(text1, mask1)
        emb2 = self.encode(text2, mask2)
        
        # Compute similarity
        combined = torch.cat([emb1, emb2], dim=1)
        similarity = self.similarity_head(combined)
        
        return similarity, emb1, emb2
    
    def score_concepts(
        self,
        text_embedding: torch.Tensor,
        concept_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Score text against multiple concepts.
        
        Args:
            text_embedding: Encoded text [batch, features]
            concept_embeddings: Concept embeddings [batch, num_concepts, features]
        
        Returns:
            scores: Concept scores [batch, num_concepts]
        """
        batch_size, num_concepts, _ = concept_embeddings.shape
        text_expanded = text_embedding.unsqueeze(1).expand(-1, num_concepts, -1)
        
        # Score each concept
        concept_scores = []
        for i in range(num_concepts):
            concept_emb = concept_embeddings[:, i, :]
            
            # Use similarity head for each concept
            combined = torch.cat([text_embedding, concept_emb], dim=1)
            score = self.similarity_head(combined)
            concept_scores.append(score)
        
        return torch.cat(concept_scores, dim=1)


class SemanticDataset(Dataset):
    """Dataset for semantic similarity training."""
    
    def __init__(self, embeddings1: np.ndarray, embeddings2: np.ndarray, labels: np.ndarray):
        self.embeddings1 = torch.FloatTensor(embeddings1)
        self.embeddings2 = torch.FloatTensor(embeddings2)
        self.labels = torch.FloatTensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return {
            'emb1': self.embeddings1[idx],
            'emb2': self.embeddings2[idx],
            'label': self.labels[idx]
        }


class NNSemanticEngine:
    """
    Pure neural network engine for semantic analysis.
    Works with pre-computed embeddings.
    """
    
    def __init__(self, config: Optional[NNConfig] = None):
        self.config = config or NNConfig()
        self.device = torch.device(self.config.device)
        
        # Initialize model
        self.model = SemanticNN(self.config).to(self.device)
        
        # Optimizer and loss
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=1e-4
        )
        self.criterion = nn.MSELoss()
        
        # Training history
        self.history = {'train_loss': [], 'val_loss': []}
        
        logger.info(f"Initialized NNSemanticEngine on {self.device}")
    
    def train_epoch(
        self, 
        train_loader: DataLoader,
        progress_bar: bool = True
    ) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        iterator = tqdm(train_loader, desc="Training") if progress_bar else train_loader
        
        for batch in iterator:
            emb1 = batch['emb1'].to(self.device)
            emb2 = batch['emb2'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward pass
            similarity, _, _ = self.model(emb1, emb2)
            loss = self.criterion(similarity.squeeze(), labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if progress_bar:
                iterator.set_postfix({'loss': loss.item()})
        
        return total_loss / max(num_batches, 1)
    
    def validate(
        self,
        val_loader: DataLoader,
        progress_bar: bool = True
    ) -> float:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        iterator = tqdm(val_loader, desc="Validation") if progress_bar else val_loader
        
        with torch.no_grad():
            for batch in iterator:
                emb1 = batch['emb1'].to(self.device)
                emb2 = batch['emb2'].to(self.device)
                labels = batch['label'].to(self.device)
                
                similarity, _, _ = self.model(emb1, emb2)
                loss = self.criterion(similarity.squeeze(), labels)
                
                total_loss += loss.item()
                num_batches += 1
                
                if progress_bar:
                    iterator.set_postfix({'loss': loss.item()})
        
        return total_loss / max(num_batches, 1)
    
    def fit(
        self,
        train_embeddings1: np.ndarray,
        train_embeddings2: np.ndarray,
        train_labels: np.ndarray,
        val_embeddings1: Optional[np.ndarray] = None,
        val_embeddings2: Optional[np.ndarray] = None,
        val_labels: Optional[np.ndarray] = None,
        epochs: int = 20,
        early_stopping_patience: int = 5
    ):
        """
        Train the neural network.
        
        Args:
            train_embeddings1: First text embeddings for training
            train_embeddings2: Second text embeddings for training
            train_labels: Similarity labels (0-1)
            val_embeddings1: Validation embeddings
            val_embeddings2: Validation embeddings
            val_labels: Validation labels
            epochs: Number of training epochs
            early_stopping_patience: Patience for early stopping
        """
        # Create datasets
        train_dataset = SemanticDataset(train_embeddings1, train_embeddings2, train_labels)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )
        
        val_loader = None
        if val_embeddings1 is not None and val_embeddings2 is not None and val_labels is not None:
            val_dataset = SemanticDataset(val_embeddings1, val_embeddings2, val_labels)
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False
            )
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            logger.info(f"Epoch {epoch + 1}/{epochs}")
            
            # Train
            train_loss = self.train_epoch(train_loader)
            self.history['train_loss'].append(train_loss)
            
            # Validate
            if val_loader:
                val_loss = self.validate(val_loader)
                self.history['val_loss'].append(val_loss)
                
                logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    torch.save(self.model.state_dict(), 'best_model.pth')
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        logger.info(f"Early stopping at epoch {epoch + 1}")
                        break
            else:
                logger.info(f"Train Loss: {train_loss:.4f}")
        
        # Load best model
        if val_loader and Path('best_model.pth').exists():
            self.model.load_state_dict(torch.load('best_model.pth'))
            logger.info("Loaded best model")
    
    def predict_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray
    ) -> float:
        """Predict similarity between two embeddings."""
        self.model.eval()
        
        with torch.no_grad():
            emb1 = torch.FloatTensor(embedding1).unsqueeze(0).to(self.device)
            emb2 = torch.FloatTensor(embedding2).unsqueeze(0).to(self.device)
            
            similarity, _, _ = self.model(emb1, emb2)
            
            return similarity.squeeze().cpu().item()
    
    def batch_predict_similarity(
        self,
        embeddings1: np.ndarray,
        embeddings2: np.ndarray
    ) -> np.ndarray:
        """Predict similarities for batch of embeddings."""
        self.model.eval()
        predictions = []
        batch_size = self.config.batch_size
        
        with torch.no_grad():
            for i in range(0, len(embeddings1), batch_size):
                batch_emb1 = torch.FloatTensor(
                    embeddings1[i:i+batch_size]
                ).to(self.device)
                
                batch_emb2 = torch.FloatTensor(
                    embeddings2[i:i+batch_size]
                ).to(self.device)
                
                similarity, _, _ = self.model(batch_emb1, batch_emb2)
                predictions.extend(similarity.squeeze().cpu().numpy())
        
        return np.array(predictions)
    
    def encode_text(
        self,
        embeddings: np.ndarray,
        is_sequence: bool = False
    ) -> np.ndarray:
        """Encode embeddings using the neural network."""
        self.model.eval()
        encoded_embeddings = []
        batch_size = self.config.batch_size
        
        with torch.no_grad():
            for i in range(0, len(embeddings), batch_size):
                batch_emb = torch.FloatTensor(
                    embeddings[i:i+batch_size]
                ).to(self.device)
                
                if is_sequence and batch_emb.dim() == 2:
                    # Add sequence dimension if needed
                    batch_emb = batch_emb.unsqueeze(1)
                
                encoded, _, _ = self.model(batch_emb, batch_emb)
                encoded_embeddings.append(encoded.cpu().numpy())
        
        return np.vstack(encoded_embeddings)
    
    def evaluate_answer(
        self,
        answer_embedding: np.ndarray,
        concept_embeddings: np.ndarray,
        threshold: float = 0.6
    ) -> Dict[str,]:
        """
        Evaluate answer against concepts.
        
        Args:
            answer_embedding: Answer embedding
            concept_embeddings: Array of concept embeddings
            threshold: Similarity threshold
        
        Returns:
            Evaluation results
        """
        self.model.eval()
        
        # Prepare tensors
        answer_tensor = torch.FloatTensor(answer_embedding).unsqueeze(0).to(self.device)
        concepts_tensor = torch.FloatTensor(concept_embeddings).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # Get answer encoding
            answer_encoded = self.model.encode(answer_tensor)
            
            # Score against each concept
            scores = []
            for i in range(concept_embeddings.shape[0]):
                concept_tensor = concepts_tensor[:, i, :]
                
                # Get concept encoding
                concept_encoded = self.model.encode(concept_tensor)
                
                # Compute similarity
                combined = torch.cat([answer_encoded, concept_encoded], dim=1)
                similarity = self.model.similarity_head(combined)
                scores.append(similarity.item())
        
        scores = np.array(scores)
        matches = scores >= threshold
        
        return {
            'scores': scores.tolist(),
            'matches': matches.tolist(),
            'average_score': float(scores.mean()),
            'coverage': float(matches.mean()) if len(matches) > 0 else 0.0,
            'matched_concepts': [i for i, m in enumerate(matches) if m]
        }
    
    def save(self, path: str):
        """Save the engine."""
        save_data = {
            'config': self.config,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'history': self.history
        }
        torch.save(save_data, path)
        logger.info(f"Engine saved to {path}")
    
    @classmethod
    def load(cls, path: str, device: Optional[str] = None):
        """Load a saved engine."""
        save_data = torch.load(path, map_location=device)
        
        # Create engine
        engine = cls(config=save_data['config'])
        
        if device:
            engine.device = torch.device(device)
            engine.model = engine.model.to(engine.device)
        
        # Load states
        engine.model.load_state_dict(save_data['model_state'])
        engine.optimizer.load_state_dict(save_data['optimizer_state'])
        engine.history = save_data['history']
        
        logger.info(f"Engine loaded from {path}")
        return engine


# Example usage
def create_simple_nn_engine(
    input_dim: int = 300,
    hidden_dims: List[int] = None,
    device: Optional[str] = None
) -> NNSemanticEngine:
    """Create a simple neural network engine."""
    config = NNConfig(
        input_dim=input_dim,
        hidden_dims=hidden_dims or [256, 128, 64],
        device=device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    return NNSemanticEngine(config)


# Utility for embedding conversion (if needed)
class EmbeddingConverter:
    """Convert text to embeddings using simple methods."""
    
    @staticmethod
    def text_to_bow(text: str, vocab: Dict[str, int], vector_size: int = 300) -> np.ndarray:
        """Convert text to bag-of-words embedding."""
        import re
        from collections import Counter
        
        words = re.findall(r'\w+', text.lower())
        word_counts = Counter(words)
        
        vector = np.zeros(vector_size)
        for word, count in word_counts.items():
            if word in vocab:
                idx = vocab[word] % vector_size
                vector[idx] += count
        
        # Normalize
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return vector
    
    @staticmethod
    def create_vocab(texts: List[str], max_vocab_size: int = 10000) -> Dict[str, int]:
        """Create vocabulary from texts."""
        from collections import Counter
        import re
        
        word_counts = Counter()
        for text in texts:
            words = re.findall(r'\w+', text.lower())
            word_counts.update(words)
        
        # Get most common words
        common_words = word_counts.most_common(max_vocab_size)
        vocab = {word: idx for idx, (word, _) in enumerate(common_words)}
        
        return vocab