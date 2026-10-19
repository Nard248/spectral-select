import torch
import torch.nn as nn



class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return x + self.fn(x)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()

        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(
            self.norm(x)
        )


class FeedForward(nn.Module):
    def __init__(
        self,
        dim,
        hidden_dim,
        dropout
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(
                dim,
                hidden_dim
            ),
            nn.GELU(),
            nn.Dropout(
                dropout
            ),
            nn.Linear(
                hidden_dim,
                dim
            ),
            nn.Dropout(
                dropout
            )
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads,
        dim_head,
        dropout
    ):
        super().__init__()

        inner_dim = (
            heads * dim_head
        )

        self.heads = heads
        self.scale = (
            dim_head ** -0.5
        )

        self.to_qkv = nn.Linear(
            dim,
            inner_dim * 3,
            bias=False
        )

        self.to_out = nn.Sequential(
            nn.Linear(
                inner_dim,
                dim
            ),
            nn.Dropout(
                dropout
            )
        )

    def forward(self, x):

        q, k, v = (
            self.to_qkv(x)
            .chunk(
                3,
                dim=-1
            )
        )

        b, n, _ = q.shape

        q, k, v = map(
            lambda t: t.reshape(
                b,
                n,
                self.heads,
                -1
            ).permute(
                0,
                2,
                1,
                3
            ),
            (q, k, v)
        )

        attn = (
            torch.einsum(
                "bhid,bhjd->bhij",
                q,
                k
            )
            * self.scale
        )

        attn = attn.softmax(
            dim=-1
        )

        out = torch.einsum(
            "bhij,bhjd->bhid",
            attn,
            v
        )

        out = out.permute(
            0,
            2,
            1,
            3
        ).reshape(
            b,
            n,
            -1
        )

        return self.to_out(
            out
        )


class SmartTransformer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout,
        num_tokens
    ):
        super().__init__()

        self.layers = nn.ModuleList([
            nn.ModuleList([
                Residual(
                    PreNorm(
                        dim,
                        Attention(
                            dim=dim,
                            heads=heads,
                            dim_head=dim_head,
                            dropout=dropout
                        )
                    )
                ),

                Residual(
                    PreNorm(
                        dim,
                        FeedForward(
                            dim=dim,
                            hidden_dim=mlp_dim,
                            dropout=dropout
                        )
                    )
                )
            ])

            for _ in range(depth)
        ])

        # Cross-layer feature fusion
        self.skipcat = nn.ModuleList([
            nn.Conv2d(
                num_tokens + 1,
                num_tokens + 1,
                kernel_size=(1, 2),
                stride=1,
                padding=0
            )

            for _ in range(
                max(
                    0,
                    depth - 2
                )
            )
        ])

    def forward(self, x):

        previous = []

        for i, (
            attention,
            ffn
        ) in enumerate(
            self.layers
        ):

            previous.append(x)

            if i > 1:

                fusion = torch.cat(
                    [
                        x.unsqueeze(-1),

                        previous[
                            i - 2
                        ].unsqueeze(-1)
                    ],
                    dim=-1
                )

                x = self.skipcat[
                    i - 2
                ](
                    fusion
                ).squeeze(-1)

            x = attention(x)
            x = ffn(x)

        return x


class MESmartAttention(nn.Module):
    """
    ME SmartAttention

    Input:
        [batch, excitations, emissions]

    For each emission position, the token contains:
        previous emission across all excitations
        current emission across all excitations
        next emission across all excitations

    edge_mode:

        "circular"
            E1  -> [E30, E1, E2]
            E30 -> [E29, E30, E1]

        "zero"
            E1  -> [0, E1, E2]
            E30 -> [E29, E30, 0]

        "discard"
            No token is created for E1 or E30.

            E2  -> [E1, E2, E3]
            ...
            E29 -> [E28, E29, E30]

            This produces n_emissions - 2 tokens.
    """

    def __init__(
        self,
        n_excitations,
        n_emissions,
        n_classes,

        dim=16,
        depth=3,
        heads=2,
        dim_head=8,
        mlp_dim=4,

        dropout=0.10,
        embedding_dropout=0.10,

        edge_mode="zero"
    ):
        super().__init__()

        if edge_mode not in {
            "circular",
            "zero",
            "discard"
        }:
            raise ValueError(
                "edge_mode must be "
                "'circular', 'zero', or 'discard'"
            )

        if n_emissions < 3:
            raise ValueError(
                "At least 3 emission positions are required."
            )

        self.edge_mode = edge_mode
        self.n_emissions = n_emissions

        if edge_mode == "discard":
            self.num_tokens = (
                n_emissions - 2
            )
        else:
            self.num_tokens = (
                n_emissions
            )

        token_dim = (
            3 * n_excitations
        )

        # Raw neighborhood token -> embedding
        self.token_embedding = nn.Linear(
            token_dim,
            dim
        )

        # Keep positional embeddings for the full
        # original emission axis.
        #
        # index 0 = CLS
        # index 1 = emission 1
        # ...
        # index N = emission N
        self.pos_embedding = nn.Parameter(
            torch.randn(
                1,
                n_emissions + 1,
                dim
            ) * 0.02
        )

        self.cls_token = nn.Parameter(
            torch.randn(
                1,
                1,
                dim
            ) * 0.02
        )

        self.dropout = nn.Dropout(
            embedding_dropout
        )

        self.transformer = SmartTransformer(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=dropout,
            num_tokens=self.num_tokens
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(
                dim
            ),

            nn.Linear(
                dim,
                n_classes
            )
        )


    def make_tokens(self, x):
        """
        Input:
            [B, excitation, emission]

        Output:
            circular/zero:
                [B, emissions, 3 * excitations]

            discard:
                [B, emissions - 2, 3 * excitations]
        """

        # [B, excitation, emission]
        # ->
        # [B, emission, excitation]

        x = x.transpose(
            1,
            2
        )


        # ====================================================
        # CIRCULAR
        # ====================================================

        if self.edge_mode == "circular":

            previous = torch.roll(
                x,
                shifts=1,
                dims=1
            )

            following = torch.roll(
                x,
                shifts=-1,
                dims=1
            )

            tokens = torch.cat(
                [
                    previous,
                    x,
                    following
                ],
                dim=-1
            )


        # ====================================================
        # ZERO PADDING
        # ====================================================

        elif self.edge_mode == "zero":

            zero = torch.zeros_like(
                x[:, :1]
            )

            previous = torch.cat(
                [
                    zero,
                    x[:, :-1]
                ],
                dim=1
            )

            following = torch.cat(
                [
                    x[:, 1:],
                    zero
                ],
                dim=1
            )

            tokens = torch.cat(
                [
                    previous,
                    x,
                    following
                ],
                dim=-1
            )


        # ====================================================
        # DISCARD EDGE TOKENS
        #
        # Centers are emissions 2 ... N-1.
        #
        # previous = E1 ... E(N-2)
        # current  = E2 ... E(N-1)
        # next     = E3 ... EN
        # ====================================================

        else:

            previous = x[
                :, :-2
            ]

            current = x[
                :, 1:-1
            ]

            following = x[
                :, 2:
            ]

            tokens = torch.cat(
                [
                    previous,
                    current,
                    following
                ],
                dim=-1
            )


        return tokens


    def get_position_embeddings(self):
        """
        Return positional embeddings matching the
        selected edge mode.
        """

        if self.edge_mode == "discard":

            # CLS position
            cls_position = self.pos_embedding[
                :, :1
            ]

            # Original emission positions 2 ... N-1
            emission_positions = self.pos_embedding[
                :,
                2:self.n_emissions
            ]

            return torch.cat(
                [
                    cls_position,
                    emission_positions
                ],
                dim=1
            )

        return self.pos_embedding


    def forward(self, x):

        # Build local emission-neighborhood tokens
        x = self.make_tokens(x)

        # Raw neighborhood -> DIM embedding
        x = self.token_embedding(x)

        # CLS token
        cls = self.cls_token.expand(
            x.shape[0],
            -1,
            -1
        )

        x = torch.cat(
            [
                cls,
                x
            ],
            dim=1
        )

        # Add positional information
        position_embeddings = (
            self.get_position_embeddings()
        )

        x = (
            x
            + position_embeddings
        )

        x = self.dropout(x)

        # Transformer
        x = self.transformer(x)

        # CLS -> logits
        return self.classifier(
            x[:, 0]
        )