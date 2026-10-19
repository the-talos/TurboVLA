"""Device selection for the inference path: XPU, CUDA and CPU.

Everything except the last class runs on any host (availability is mocked), so a
CUDA-only CI stays green. The parity test builds a tiny random-weight TurboVLA
from config (tiny BERT and DINOv3 written to a temporary directory) and compares
CPU against XPU; it is skipped when no XPU is present.
"""

import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

import torch

from turbovla import device as dev
from turbovla.models import text_encoder, vision_encoder
from turbovla.models.configuration import TurboVLAConfig


def _patch_availability(*, xpu: bool, cuda: bool):
    def fake_is_available(device_type: str) -> bool:
        if device_type == "xpu":
            return xpu
        if device_type == "cuda":
            return cuda
        return device_type == "cpu"

    return mock.patch.object(dev, "is_available", side_effect=fake_is_available)


class DeviceDetectionTest(unittest.TestCase):
    def test_xpu_preferred_over_cuda(self):
        with _patch_availability(xpu=True, cuda=True):
            self.assertEqual(dev.default_device_type(), "xpu")
            self.assertEqual(dev.resolve_device(None), torch.device("xpu"))

    def test_cuda_when_no_xpu(self):
        with _patch_availability(xpu=False, cuda=True):
            self.assertEqual(dev.resolve_device(None), torch.device("cuda"))

    def test_cpu_when_no_accelerator(self):
        with _patch_availability(xpu=False, cuda=False):
            self.assertEqual(dev.resolve_device(None), torch.device("cpu"))

    def test_explicit_device_wins_over_detection(self):
        with _patch_availability(xpu=True, cuda=True):
            self.assertEqual(dev.resolve_device("cuda:1"), torch.device("cuda:1"))
            self.assertEqual(dev.resolve_device(torch.device("cpu")), torch.device("cpu"))

    def test_seed_reaches_each_available_backend(self):
        with _patch_availability(xpu=True, cuda=True), mock.patch.object(
            torch.cuda, "manual_seed_all", create=True
        ) as cuda_seed, mock.patch.object(torch.xpu, "manual_seed_all", create=True) as xpu_seed:
            dev.seed_accelerators(5)
        cuda_seed.assert_called_once_with(5)
        xpu_seed.assert_called_once_with(5)

    def test_seed_skips_missing_backends(self):
        with _patch_availability(xpu=False, cuda=False), mock.patch.object(
            torch.cuda, "manual_seed_all", create=True
        ) as cuda_seed:
            dev.seed_accelerators(5)
        cuda_seed.assert_not_called()


class AutocastTypeTest(unittest.TestCase):
    def test_accelerators_use_their_own_backend(self):
        self.assertEqual(dev.autocast_device_type(torch.device("xpu")), "xpu")
        self.assertEqual(dev.autocast_device_type("cuda:0"), "cuda")

    def test_cpu_has_no_autocast(self):
        self.assertIsNone(dev.autocast_device_type(torch.device("cpu")))


class AttentionImplementationTest(unittest.TestCase):
    def setUp(self):
        dev._warned.clear()

    def test_flash_attention_becomes_sdpa_without_cuda(self):
        with _patch_availability(xpu=True, cuda=False), warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self.assertEqual(dev.resolve_attention_implementation("flash_attention_2"), "sdpa")
            self.assertEqual(dev.resolve_attention_implementation("flash_attention_2"), "sdpa")
        self.assertEqual(len(caught), 1, "the fallback must be logged once")

    def test_flash_attention_kept_on_cuda(self):
        with _patch_availability(xpu=False, cuda=True):
            self.assertEqual(dev.resolve_attention_implementation("flash_attention_2"), "flash_attention_2")

    def test_other_values_untouched(self):
        with _patch_availability(xpu=True, cuda=False):
            self.assertEqual(dev.resolve_attention_implementation("eager"), "eager")
            self.assertIsNone(dev.resolve_attention_implementation(None))

    def test_encoders_request_sdpa_without_cuda(self):
        for module, config_cls in (
            (text_encoder, text_encoder.TextEncoderConfig),
            (vision_encoder, vision_encoder.VisionEncoderConfig),
        ):
            config = config_cls(attention_implementation="flash_attention_2")
            with _patch_availability(xpu=True, cuda=False), mock.patch.object(
                module.AutoModel, "from_pretrained", return_value="model"
            ) as load, warnings.catch_warnings():
                warnings.simplefilter("ignore")
                module._load_pretrained_model(config)
            self.assertEqual(load.call_args.kwargs["attn_implementation"], "sdpa")


def _write_tiny_assets(root: Path) -> tuple[str, str]:
    from transformers import BertConfig, BertModel, BertTokenizer, DINOv3ViTConfig, DINOv3ViTModel

    bert_dir, dino_dir = root / "bert", root / "dino"
    bert_dir.mkdir()
    words = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", ".", "?", "pick", "up", "the", "red", "cup"]
    (bert_dir / "vocab.txt").write_text("\n".join(words) + "\n")
    BertTokenizer(str(bert_dir / "vocab.txt")).save_pretrained(bert_dir)
    BertModel(
        BertConfig(
            vocab_size=len(words), hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64
        )
    ).save_pretrained(bert_dir)
    DINOv3ViTModel(
        DINOv3ViTConfig(
            image_size=32,
            patch_size=16,
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=64,
            num_register_tokens=0,
        )
    ).save_pretrained(dino_dir)
    return str(bert_dir), str(dino_dir)


@unittest.skipIf(not dev.is_available("xpu"), "XPU not available")
class XpuParityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from turbovla.models import build_turbovla

        cls._tmp = tempfile.TemporaryDirectory()
        bert_path, dino_path = _write_tiny_assets(Path(cls._tmp.name))
        config = TurboVLAConfig()
        config.text.model_name_or_path = bert_path
        config.vision.model_name_or_path = dino_path
        config.vision.image_size = 32
        config.vision.compute_precision = "fp32"
        config.interaction.hidden_dim = 32
        config.interaction.nheads = 2
        config.interaction.num_layers = 1
        config.interaction.dim_feedforward = 64
        config.interaction.enhancer_inner_dim = 64
        config.action.num_layers = 1
        config.action.mlp_hidden_dim = 32
        config.action.state_hidden_dim = 32
        torch.manual_seed(0)
        cls.model = build_turbovla(config).eval()
        # Re-randomise every parameter so that zero-initialised layers cannot hide a mismatch.
        with torch.no_grad():
            for param in cls.model.parameters():
                param.copy_(torch.randn_like(param) * 0.1)
        cls.samples = torch.randn(1, 2, 3, 32, 32)
        cls.state = torch.randn(1, 8)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def _predict(self, device: str, dtype: torch.dtype) -> torch.Tensor:
        model = type(self).model.to(device=device, dtype=dtype)
        with torch.inference_mode():
            out = model(["pick up the red cup."], self.samples.to(device, dtype), self.state.to(device, dtype))
        return out.float().cpu()

    def test_float32_actions_match_cpu(self):
        cpu = self._predict("cpu", torch.float32)
        xpu = self._predict("xpu", torch.float32)
        self.assertEqual(cpu.shape, xpu.shape)
        torch.testing.assert_close(xpu, cpu, atol=1e-3, rtol=1e-3)

    def test_bfloat16_actions_match_cpu(self):
        cpu = self._predict("cpu", torch.bfloat16)
        xpu = self._predict("xpu", torch.bfloat16)
        torch.testing.assert_close(xpu, cpu, atol=5e-2, rtol=5e-2)


if __name__ == "__main__":
    unittest.main()
