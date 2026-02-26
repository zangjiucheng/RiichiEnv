"""Sequence feature encoder for transformer models.

Based on the Kanachan v3 encoding design. Wraps the Rust sequence feature
encoding methods on Observation, producing padded tensors with masks
suitable for batched training.

See docs/SEQUENCE_FEATURE_ENCODING.md for the full specification.
"""

import numpy as np
import torch


class SequenceFeatureEncoder:
    """Sequence feature encoder for transformer models.

    Produces:
        sparse:      (MAX_SPARSE_LEN,)   int64   padded sparse embedding indices
        numeric:     (NUM_NUMERIC,)       float32
        progression: (MAX_PROG_LEN, 5)   int64   padded action-history 5-tuples
        candidates:  (MAX_CAND_LEN, 4)   int64   padded legal-action 4-tuples
        sparse_mask: (MAX_SPARSE_LEN,)   bool    True for real tokens
        prog_mask:   (MAX_PROG_LEN,)     bool    True for real entries
        cand_mask:   (MAX_CAND_LEN,)     bool    True for real entries
    """

    SPARSE_VOCAB_SIZE = 442
    SPARSE_PAD = 441
    MAX_SPARSE_LEN = 25

    PROG_DIMS = (5, 277, 3, 3, 5)
    PROG_PAD = (4, 276, 2, 2, 4)
    MAX_PROG_LEN = 512

    CAND_DIMS = (280, 3, 3, 4)
    CAND_PAD = (279, 2, 2, 3)
    MAX_CAND_LEN = 64

    NUM_NUMERIC = 12

    def __init__(self, n_players: int = 4, game_style: int = 1):
        self.n_players = n_players
        self.game_style = game_style  # 0=tonpuusen, 1=hanchan

    def encode(self, obs) -> dict[str, torch.Tensor]:
        """Encode observation into sequence features for transformer models.

        Args:
            obs: riichienv Observation object with encode_seq_* methods.

        Returns:
            Dict with keys: sparse, numeric, progression, candidates,
                           sparse_mask, prog_mask, cand_mask
        """
        # Sparse
        raw = np.frombuffer(
            obs.encode_seq_sparse(self.game_style), dtype=np.uint16
        ).copy()
        n_sparse = min(len(raw), self.MAX_SPARSE_LEN)
        sparse = np.full(self.MAX_SPARSE_LEN, self.SPARSE_PAD, dtype=np.int64)
        sparse[:n_sparse] = raw[:n_sparse]
        sparse_mask = np.zeros(self.MAX_SPARSE_LEN, dtype=np.bool_)
        sparse_mask[:n_sparse] = True

        # Numeric
        numeric = np.frombuffer(
            obs.encode_seq_numeric(), dtype=np.float32
        ).copy()

        # Progression
        prog_bytes = obs.encode_seq_progression()
        if len(prog_bytes) > 0:
            raw_prog = np.frombuffer(prog_bytes, dtype=np.uint16).reshape(-1, 5)
            n_prog = min(len(raw_prog), self.MAX_PROG_LEN)
        else:
            raw_prog = np.empty((0, 5), dtype=np.uint16)
            n_prog = 0
        prog = np.tile(
            np.array(self.PROG_PAD, dtype=np.int64), (self.MAX_PROG_LEN, 1)
        )
        if n_prog > 0:
            prog[:n_prog] = raw_prog[:n_prog]
        prog_mask = np.zeros(self.MAX_PROG_LEN, dtype=np.bool_)
        prog_mask[:n_prog] = True

        # Candidates
        cand_bytes = obs.encode_seq_candidates()
        if len(cand_bytes) > 0:
            raw_cand = np.frombuffer(cand_bytes, dtype=np.uint16).reshape(-1, 4)
            n_cand = min(len(raw_cand), self.MAX_CAND_LEN)
        else:
            raw_cand = np.empty((0, 4), dtype=np.uint16)
            n_cand = 0
        cand = np.tile(
            np.array(self.CAND_PAD, dtype=np.int64), (self.MAX_CAND_LEN, 1)
        )
        if n_cand > 0:
            cand[:n_cand] = raw_cand[:n_cand]
        cand_mask = np.zeros(self.MAX_CAND_LEN, dtype=np.bool_)
        cand_mask[:n_cand] = True

        return {
            "sparse": torch.from_numpy(sparse),
            "numeric": torch.from_numpy(numeric),
            "progression": torch.from_numpy(prog),
            "candidates": torch.from_numpy(cand),
            "sparse_mask": torch.from_numpy(sparse_mask),
            "prog_mask": torch.from_numpy(prog_mask),
            "cand_mask": torch.from_numpy(cand_mask),
        }


class SequenceFeaturePackedEncoder:
    """Packed single-tensor encoder for Ray worker compatibility.

    Packs all sequence features into a flat float32 tensor so the teacher
    worker (which expects ``encoder.encode(obs) -> Tensor``) can handle it
    transparently.  The ``TransformerActorCritic`` model unpacks this
    internally.

    Layout (all float32):
        sparse      (25)       int indices stored as float
        numeric     (12)       continuous values
        progression (512 * 5)  int tuples stored as float
        candidates  (64 * 4)   int tuples stored as float
        sparse_mask (25)       bool stored as float
        prog_mask   (512)      bool stored as float
        cand_mask   (64)       bool stored as float
        ─────────────────────
        Total: 3454 float32
    """

    _S = SequenceFeatureEncoder.MAX_SPARSE_LEN   # 25
    _N = SequenceFeatureEncoder.NUM_NUMERIC       # 12
    _P = SequenceFeatureEncoder.MAX_PROG_LEN     # 512
    _C = SequenceFeatureEncoder.MAX_CAND_LEN     # 64

    PACKED_SIZE = _S + _N + _P * 5 + _C * 4 + _S + _P + _C  # 3454

    def __init__(self, tile_dim: int = 34, n_players: int = 4,
                 game_style: int = 1):
        # tile_dim accepted for API compatibility with CNN encoders
        if tile_dim == 27:
            n_players = 3
        self.inner = SequenceFeatureEncoder(
            n_players=n_players, game_style=game_style)

    def encode(self, obs) -> torch.Tensor:
        """Encode observation into a flat packed tensor (3454,)."""
        d = self.inner.encode(obs)
        packed = torch.zeros(self.PACKED_SIZE, dtype=torch.float32)
        o = 0

        packed[o:o + self._S] = d["sparse"].float();           o += self._S
        packed[o:o + self._N] = d["numeric"];                   o += self._N
        packed[o:o + self._P * 5] = d["progression"].reshape(-1).float()
        o += self._P * 5
        packed[o:o + self._C * 4] = d["candidates"].reshape(-1).float()
        o += self._C * 4
        packed[o:o + self._S] = d["sparse_mask"].float();      o += self._S
        packed[o:o + self._P] = d["prog_mask"].float();        o += self._P
        packed[o:o + self._C] = d["cand_mask"].float()

        return packed
