import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def get_device():
    return torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def scaled_dot_product(q, k, v, mask=None):
    d_k = q.size()[-1] # 30 X 8 X 200 X 64
    scaled = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d_k)# 30 X 8 X 200 X 200
    if mask is not None:
        scaled = scaled.permute(1, 0, 2, 3) + mask # 8 X 30 X 200 X 200, doing this because we are using mask with tesnsor 8 size
        scaled = scaled.permute(1, 0, 2, 3)
    attention = F.softmax(scaled, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv_layer = nn.Linear(d_model , 3 * d_model)
        self.linear_layer = nn.Linear(d_model, d_model)

    def forward(self, x, mask):
        batch_size, sequence_length, d_model = x.size() # 30 X 200 X 512
        qkv = self.qkv_layer(x) # 30 X 200 X 512
        qkv = qkv.reshape(batch_size, sequence_length, self.num_heads, 3 * self.head_dim) # 30 X 200 X 8 X 64
        qkv = qkv.permute(0, 2, 1, 3) # 30 X 8 X 200 X 64
        q, k, v = qkv.chunk(3, dim=-1) # 30 X 8 X 200 X 64 each
        values, attention = scaled_dot_product(q, k, v, mask)
        values = values.permute(0, 2, 1, 3).reshape(batch_size, sequence_length, self.num_heads * self.head_dim)
        out = self.linear_layer(values)
        return out

class LayerNormalization(nn.Module):
    def __init__(self, parameters_shape, eps=1e-5):
        super().__init__()
        self.parameters_shape=parameters_shape
        self.eps=eps
        self.gamma = nn.Parameter(torch.ones(parameters_shape))
        self.beta =  nn.Parameter(torch.zeros(parameters_shape))

    def forward(self, inputs):
        dims = [-(i + 1) for i in range(len(self.parameters_shape))]
        mean = inputs.mean(dim=dims, keepdim=True)
        var = ((inputs - mean) ** 2).mean(dim=dims, keepdim=True)
        std = (var + self.eps).sqrt()
        y = (inputs - mean) / std
        out = self.gamma * y + self.beta
        return out
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_sequence_length):
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.d_model = d_model

    def forward(self):
        even_i = torch.arange(0, self.d_model, 2).float()
        denominator = torch.pow(10000, even_i/self.d_model)
        position = (torch.arange(self.max_sequence_length)
                          .reshape(self.max_sequence_length, 1))
        even_PE = torch.sin(position / denominator)
        odd_PE = torch.cos(position / denominator)
        stacked = torch.stack([even_PE, odd_PE], dim=2)
        PE = torch.flatten(stacked, start_dim=1, end_dim=2)
        return PE

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
    
class SentenceEmbedding(nn.Module):
    "For a given sentence, create an embedding"
    def __init__(self, max_sequence_length, d_model, language_to_index, START_TOKEN, END_TOKEN, PADDING_TOKEN):
        super().__init__()
        self.language_to_index = language_to_index
        self.START_TOKEN = START_TOKEN
        self.END_TOKEN = END_TOKEN
        self.PADDING_TOKEN = PADDING_TOKEN

        # Ensure special tokens exist in vocabulary
        self._ensure_special_tokens()

        # Define vocab size
        self.vocab_size = len(self.language_to_index)
        self.max_sequence_length = max_sequence_length
        self.embedding = nn.Embedding(self.vocab_size, d_model)
        self.position_encoder = PositionalEncoding(d_model, max_sequence_length)
        self.dropout = nn.Dropout(p=0.1)

    def _ensure_special_tokens(self):
        """Ensure special tokens exist in language_to_index"""
        special_tokens = [self.START_TOKEN, self.END_TOKEN, self.PADDING_TOKEN]
        for token in special_tokens:
            if token not in self.language_to_index:
                self.language_to_index[token] = len(self.language_to_index)  # Assign new index

    def batch_tokenize(self, batch, start_token, end_token):
        def tokenize(sentence, start_token, end_token):
            sentence_word_indices = []
            for token in list(sentence):
                if token not in self.language_to_index:
                    raise ValueError(f"Unknown token '{token}' found! Update your vocabulary.")
                sentence_word_indices.append(self.language_to_index[token])

            # Add start and end tokens if required
            if start_token:
                sentence_word_indices.insert(0, self.language_to_index[self.START_TOKEN])
            if end_token:
                sentence_word_indices.append(self.language_to_index[self.END_TOKEN])

            # Pad sentence to max_sequence_length
            while len(sentence_word_indices) < self.max_sequence_length:
                sentence_word_indices.append(self.language_to_index[self.PADDING_TOKEN])

            # Convert to tensor and ensure it stays within vocab range
            token_tensor = torch.tensor(sentence_word_indices)
            return torch.clamp(token_tensor, max=self.vocab_size - 1)  # Prevent out-of-range errors

        tokenized = [tokenize(batch[i], start_token, end_token) for i in range(len(batch))]
        return torch.stack(tokenized).to(get_device())

    def forward(self, x, start_token, end_token):
        x = self.batch_tokenize(x, start_token, end_token)
        x = self.embedding(x)
        pos = self.position_encoder().to(get_device())  # Ensure positional encoding gets input
        x = self.dropout(x + pos)
        return x