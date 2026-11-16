"""cli_faces: the endpoint's import target exists and the payload has the shape the
family clients read (box, mouth, embedding), with a fake insightface app (no model)."""
import os
import sys
import types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from PIL import Image
import cz_face


class _Face:
    def __init__(self):
        self.bbox = np.array([10.0, 20.0, 110.0, 140.0])
        self.kps = np.array([[30, 50], [80, 50], [55, 80], [40, 110], [70, 110]], dtype=float)
        self.normed_embedding = np.ones(512, dtype=np.float32) / 512 ** 0.5


def test_detect_faces_full_shape(monkeypatch):
    fake = types.SimpleNamespace(get=lambda arr: [_Face()])
    monkeypatch.setattr(cz_face, "_ensure_face_detector", lambda: fake)
    out = cz_face.detect_faces_full(Image.new("RGB", (200, 200)))
    assert len(out) == 1
    f = out[0]
    assert f["box"] == (10.0, 20.0, 110.0, 140.0)
    assert f["mouth"] == (55.0, 110.0)
    assert len(f["embedding"]) == 512 and abs(sum(x * x for x in f["embedding"]) - 1) < 1e-4


def test_no_face_is_an_empty_list(monkeypatch):
    monkeypatch.setattr(cz_face, "_ensure_face_detector",
                        lambda: types.SimpleNamespace(get=lambda arr: []))
    assert cz_face.detect_faces_full(Image.new("RGB", (8, 8))) == []
