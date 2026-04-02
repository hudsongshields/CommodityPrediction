import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dim:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.SiLU())
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ReturnPrediction(nn.Module):
    """
    Return predictor that uses diffusion scores as features and an LSTM head.
    Season embeddings are added only at the return head input.
    """
    def __init__(
        self,
        diffusion,
        feature_dim,
        lstm_hidden,
        mlp_hidden,
        output_dim,
        num_seasons=4,
        season_embed_dim=8,
        use_season=True,
        use_raw=False,
        t_score=0.5,
        lstm_layers=1,
        dropout=0.0,
        freeze_diffusion=True,
    ):
        super().__init__()
        self.diffusion = diffusion
        self.use_season = use_season
        self.use_raw = use_raw
        self.t_score = t_score

        if freeze_diffusion:
            for p in self.diffusion.parameters():
                p.requires_grad = False

        score_mult = len(t_score) if isinstance(t_score, (list, tuple)) else 1
        input_dim = feature_dim * (score_mult + (1 if use_raw else 0))
        if use_season:
            self.season_embedding = nn.Embedding(num_seasons, season_embed_dim)
            input_dim += season_embed_dim
        else:
            self.season_embedding = None

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.head = MLP(lstm_hidden, mlp_hidden, output_dim)

    def _score_features(self, x, t_score):
        B, L, D = x.shape
        x_flat = x.view(B * L, D)

        def score_at(t):
            if torch.is_tensor(t):
                if t.dim() == 0:
                    t_batch = torch.full((B * L,), t.item(), dtype=x.dtype, device=x.device)
                elif t.dim() == 1 and t.numel() == B:
                    t_batch = t.view(B, 1).expand(B, L).reshape(B * L).to(x.device).to(x.dtype)
                else:
                    raise ValueError("t_score must be scalar or shape (B,).")
            else:
                t_batch = torch.full((B * L,), float(t), dtype=x.dtype, device=x.device)
            return self.diffusion(x_flat, t_batch).view(B, L, D)

        if isinstance(t_score, (list, tuple)):
            scores_list = [score_at(t) for t in t_score]
            return torch.cat(scores_list, dim=-1)

        return score_at(t_score)

    def forward(self, x, season_idx=None, t_score=None):
        t_score = self.t_score if t_score is None else t_score
        scores = self._score_features(x, t_score)

        feats = scores
        if self.use_raw:
            feats = torch.cat([x, scores], dim=-1)

        if self.use_season:
            if season_idx.dim() == 1:
                season_idx = season_idx.unsqueeze(1).expand(-1, x.size(1))
            season_emb = self.season_embedding(season_idx)
            feats = torch.cat([feats, season_emb], dim=-1)

        lstm_out, _ = self.lstm(feats)
        last = lstm_out[:, -1, :]
        return self.head(last)
