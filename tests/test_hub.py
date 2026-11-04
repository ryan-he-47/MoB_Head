"""Artifact integrity tests using a tiny CPU-only head and synthetic codebook."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from safetensors.torch import save_file
import torch

from mob.hub import load_head, sha256
from mob.mob_common import SwiGLUHead


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        torch.set_num_threads(2)
        self.head = SwiGLUHead(h=8, hidden=12, k=2, m=3).eval()
        save_file(self.head.state_dict(), str(self.root / "model.safetensors"))
        vertices = np.arange(5, dtype=np.int64)
        codes = ((vertices[:, None] >> np.arange(2, -1, -1)) & 1).astype(np.uint8)
        np.savez(self.root / "codebook.npz", codes=codes, vertex_id=vertices)
        self.config = {"format_version": 1, "model_type": "mob_head", "head_type": "swiglu",
                       "head_kwargs": {"h": 8, "hidden": 12, "k": 2, "m": 3}, "vocab_size": 5,
                       "sha256": {name: sha256(self.root / name)
                                  for name in ("model.safetensors", "codebook.npz")}}
        self.write_config()

    def tearDown(self):
        self.temporary.cleanup()

    def write_config(self):
        (self.root / "config.json").write_text(json.dumps(self.config), encoding="utf-8")

    def test_roundtrip_preserves_predictions(self):
        restored, codebook, _ = load_head(self.root)
        inputs = torch.randn(4, 8)
        with torch.inference_mode():
            for actual, expected in zip(restored(inputs), self.head(inputs)):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(codebook["codes"].shape, (5, 3))

    def test_corrupt_weights_rejected_before_deserialization(self):
        with (self.root / "model.safetensors").open("ab") as stream:
            stream.write(b"corruption")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            load_head(self.root)

    def test_token_order_mismatch_rejected_even_with_valid_hash(self):
        archive = np.load(self.root / "codebook.npz")
        codes, vertices = archive["codes"], archive["vertex_id"]
        archive.close()
        np.savez(self.root / "codebook.npz", codes=codes[::-1], vertex_id=vertices)
        self.config["sha256"]["codebook.npz"] = sha256(self.root / "codebook.npz")
        self.write_config()
        with self.assertRaisesRegex(ValueError, "token-ID-to-vertex"):
            load_head(self.root)

    def test_unknown_format_rejected(self):
        self.config["format_version"] = 99
        self.write_config()
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            load_head(self.root)


if __name__ == "__main__":
    unittest.main()
