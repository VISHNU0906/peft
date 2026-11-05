# Copyright 2025-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This test file is for tests specific to RandLora, since Randlora has some specific challenges due to the shared weights.
# These tests are copied from the test_vera.py file

import copy
import os
import warnings
from pathlib import Path

import pytest
import torch
from accelerate.utils.imports import is_bf16_available
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn
from transformers.pytorch_utils import Conv1D

from peft import PeftModel, RandLoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict


class MLP(nn.Module):
    def __init__(self, bias=True):
        super().__init__()
        self.relu = nn.ReLU()
        self.lin0 = nn.Linear(10, 20, bias=bias)
        self.lin1 = nn.Linear(20, 20, bias=bias)  # lin1 and lin2 have same shape
        self.lin2 = nn.Linear(20, 20, bias=bias)
        self.lin3 = nn.Linear(20, 2, bias=bias)
        self.sm = nn.LogSoftmax(dim=-1)

    def forward(self, X):
        X = self.lin0(X)
        X = self.relu(X)
        X = self.lin1(X)
        X = self.relu(X)
        X = self.lin2(X)
        X = self.relu(X)
        X = self.lin3(X)
        X = self.sm(X)
        return X


# Tests copied from the TestVera class in test_vera.py.
# Changes to the code file should be reflected here.
class TestRandLora:
    @pytest.fixture
    def mlp(self):
        torch.manual_seed(0)
        model = MLP()
        return model

    @pytest.fixture
    def mlp_same_prng(self, mlp):
        torch.manual_seed(0)

        config = RandLoraConfig(target_modules=["lin1", "lin2"], init_weights=False)
        # creates a default RandLora adapter
        peft_model = get_peft_model(mlp, config)
        config2 = RandLoraConfig(target_modules=["lin1", "lin2"], init_weights=False)
        peft_model.add_adapter("other", config2)
        return peft_model

    def test_multiple_adapters_same_prng_weights(self, mlp_same_prng):
        # we can have multiple adapters with the same prng key, in which case the weights should be shared
        assert (
            mlp_same_prng.base_model.model.lin1.randlora_A["default"]
            is mlp_same_prng.base_model.model.lin1.randlora_A["other"]
        )
        assert (
            mlp_same_prng.base_model.model.lin1.randlora_B["default"]
            is mlp_same_prng.base_model.model.lin1.randlora_B["other"]
        )
        assert (
            mlp_same_prng.base_model.model.lin2.randlora_A["default"]
            is mlp_same_prng.base_model.model.lin2.randlora_A["other"]
        )
        assert (
            mlp_same_prng.base_model.model.lin2.randlora_B["default"]
            is mlp_same_prng.base_model.model.lin2.randlora_B["other"]
        )

        input = torch.randn(5, 10)
        mlp_same_prng.set_adapter("default")
        output_default = mlp_same_prng(input)
        mlp_same_prng.set_adapter("other")
        output_other = mlp_same_prng(input)
        assert not torch.allclose(output_default, output_other, atol=1e-3, rtol=1e-3)

    def test_multiple_adapters_different_prng_raises(self):
        # we cannot have multiple adapters with different prng keys
        model = MLP()
        config = RandLoraConfig(target_modules=["lin1", "lin2"], init_weights=False)
        # creates a default RandLora adapter
        peft_model = get_peft_model(model, config)
        config2 = RandLoraConfig(target_modules=["lin1", "lin2"], init_weights=False, projection_prng_key=123)

        msg = (
            r"RandLora PRNG initialisation key must be the same for all adapters. Got config.projection_prng_key=123 but "
            r"previous config had 0"
        )
        with pytest.raises(ValueError, match=msg):
            peft_model.add_adapter("other", config2)

    def test_multiple_adapters_save_load_save_projection_false(self, mlp, tmp_path):
        # check saving and loading works with multiple adapters without saved projection weights
        torch.manual_seed(1)
        config = RandLoraConfig(target_modules=["lin1", "lin2"], init_weights=False, save_projection=False)
        # creates a default RandLora adapter
        peft_model = get_peft_model(mlp, config, adapter_name="first")
        config2 = RandLoraConfig(target_modules=["lin1", "lin2"], init_weights=False, save_projection=False)
        peft_model.add_adapter("second", config2)

        input = torch.randn(5, 10)
        peft_model.set_adapter("first")
        output_first = peft_model(input)
        peft_model.set_adapter("second")
        output_second = peft_model(input)

        # sanity check
        assert not torch.allclose(output_first, output_second, atol=1e-3, rtol=1e-3)

        save_path = tmp_path / "randlora"
        peft_model.save_pretrained(save_path)
        assert os.path.exists(save_path / "first" / "adapter_config.json")
        assert os.path.exists(save_path / "second" / "adapter_config.json")

        torch.manual_seed(0)
        mlp = MLP()
        peft_model = PeftModel.from_pretrained(mlp, save_path / "first", adapter_name="first")
        peft_model.load_adapter(save_path / "second", "second")

        peft_model.set_adapter("first")
        output_first_loaded = peft_model(input)
        peft_model.set_adapter("second")
        output_second_loaded = peft_model(input)

        assert torch.allclose(output_first, output_first_loaded, atol=1e-3, rtol=1e-3)
        assert torch.allclose(output_second, output_second_loaded, atol=1e-3, rtol=1e-3)

    def test_multiple_adapters_save_projection_false_contains_no_randlora_A_randlora_B(self, mlp, tmp_path):
        torch.manual_seed(1)
        config = RandLoraConfig(target_modules=["lin1", "lin2"], init_weights=False, save_projection=False)
        # creates a default RandLora adapter
        peft_model = get_peft_model(mlp, config, adapter_name="first")
        config2 = RandLoraConfig(target_modules=["lin1", "lin2"], init_weights=False, save_projection=False)
        peft_model.add_adapter("second", config2)

        save_path = tmp_path / "randlora"
        peft_model.save_pretrained(save_path)

        sd_default = {}
        with safe_open(save_path / "first" / "adapter_model.safetensors", framework="pt", device="cpu") as f:
            for key in f.keys():
                sd_default[key] = f.get_tensor(key)

        assert not any("randlora_A" in key for key in sd_default)
        assert not any("randlora_B" in key for key in sd_default)

        sd_other = {}
        with safe_open(save_path / "second" / "adapter_model.safetensors", framework="pt", device="cpu") as f:
            for key in f.keys():
                sd_other[key] = f.get_tensor(key)

        assert not any("randlora_A" in key for key in sd_other)
        assert not any("randlora_B" in key for key in sd_other)

    def test_randlora_A_randlora_B_share_memory(self, mlp_same_prng):
        randlora_A = mlp_same_prng.randlora_A["default"]
        randlora_B = mlp_same_prng.randlora_B["default"]

        # these tensors should share the same data
        assert randlora_A.data_ptr() == mlp_same_prng.base_model.model.lin1.randlora_A["default"].data_ptr()
        assert randlora_B.data_ptr() == mlp_same_prng.base_model.model.lin1.randlora_B["default"].data_ptr()
        assert randlora_A.data_ptr() == mlp_same_prng.base_model.model.lin2.randlora_A["default"].data_ptr()
        assert randlora_B.data_ptr() == mlp_same_prng.base_model.model.lin2.randlora_B["default"].data_ptr()
        # sanity check: these tensors shouldn't share the same data
        assert randlora_A.data_ptr() != randlora_B.data_ptr()

    def test_save_projection_true_deduplicates_shared_projections(self, mlp, tmp_path):
        # Shared RandLoRA projections should be serialized once for #3709, both to avoid safetensors alias errors
        # and to reduce checkpoint size.
        config = RandLoraConfig(target_modules=["lin0", "lin1", "lin2"], init_weights=False, save_projection=True)
        peft_model = get_peft_model(mlp, config)

        state_dict = get_peft_model_state_dict(peft_model)
        projection_tensors = {
            projection_name: [tensor for key, tensor in state_dict.items() if projection_name in key]
            for projection_name in ("randlora_A", "randlora_B")
        }
        for tensors in projection_tensors.values():
            assert tensors
            # Catch duplicate projection entries that alias the same storage as well as independent copies with
            # identical values.
            assert len({tensor.data_ptr() for tensor in tensors}) == len(tensors)
            assert all(
                not torch.equal(tensor, other)
                for index, tensor in enumerate(tensors)
                for other in tensors[index + 1 :]
            )

        # Direct serialization should succeed once the shared projections have no aliased entries in the state dict.
        save_file(state_dict, tmp_path / "adapter.safetensors")

    def test_save_projection_true_roundtrip(self, mlp, tmp_path):
        torch.manual_seed(1)
        config = RandLoraConfig(target_modules=["lin1", "lin2"], init_weights=False, save_projection=True)
        peft_model = get_peft_model(mlp, config)
        peft_model.eval()

        inputs = torch.randn(5, 10)
        output = peft_model(inputs)
        peft_model.save_pretrained(tmp_path)

        torch.manual_seed(0)
        with warnings.catch_warnings(record=True) as caught_warnings:
            warnings.simplefilter("always")
            loaded_model = PeftModel.from_pretrained(MLP(), tmp_path)
        assert not any("Found missing adapter keys" in str(warning.message) for warning in caught_warnings)
        loaded_model.eval()
        torch.testing.assert_close(loaded_model(inputs), output)

    def test_load_projection_true_with_duplicated_aliases(self, mlp):
        # This recreates the pre-#3709 checkpoint layout with duplicated layer-level aliases. It complements the
        # previous test by verifying that existing checkpoints remain loadable after the compact format is introduced.
        torch.manual_seed(1)
        config = RandLoraConfig(target_modules=["lin1", "lin2"], init_weights=False, save_projection=True)
        source_model = get_peft_model(mlp, config)
        source_model.eval()

        inputs = torch.randn(5, 10)
        expected_output = source_model(inputs)
        old_state_dict = {
            key.removesuffix(".default"): value.clone()
            for key, value in source_model.state_dict().items()
            if "randlora_" in key and key.endswith(".default")
        }
        # Confirm that this state dict represents the legacy format with multiple aliases for each shared projection.
        for projection_name in ("randlora_A", "randlora_B"):
            projection_values = [value for key, value in old_state_dict.items() if projection_name in key]
            assert len(projection_values) > 1
            assert all(torch.equal(projection_values[0], value) for value in projection_values[1:])

        torch.manual_seed(0)
        loaded_model = get_peft_model(MLP(), config)
        load_result = set_peft_model_state_dict(loaded_model, old_state_dict)

        assert not [key for key in load_result.missing_keys if "randlora_" in key]
        assert not load_result.unexpected_keys
        loaded_model.eval()
        torch.testing.assert_close(loaded_model(inputs), expected_output)

    def test_randlora_lambda_dont_share_memory(self, mlp_same_prng):
        # sanity check: these tensors shouldn't share the same data
        assert (
            mlp_same_prng.base_model.model.lin1.randlora_lambda["default"].data_ptr()
            != mlp_same_prng.base_model.model.lin1.randlora_lambda["other"].data_ptr()
        )
        assert (
            mlp_same_prng.base_model.model.lin1.randlora_lambda["default"].data_ptr()
            != mlp_same_prng.base_model.model.lin2.randlora_lambda["default"].data_ptr()
        )
        assert (
            mlp_same_prng.base_model.model.lin1.randlora_lambda["other"].data_ptr()
            != mlp_same_prng.base_model.model.lin2.randlora_lambda["other"].data_ptr()
        )
        assert (
            mlp_same_prng.base_model.model.lin1.randlora_gamma["default"].data_ptr()
            != mlp_same_prng.base_model.model.lin1.randlora_gamma["other"].data_ptr()
        )
        assert (
            mlp_same_prng.base_model.model.lin1.randlora_gamma["default"].data_ptr()
            != mlp_same_prng.base_model.model.lin2.randlora_gamma["default"].data_ptr()
        )
        assert (
            mlp_same_prng.base_model.model.lin1.randlora_gamma["other"].data_ptr()
            != mlp_same_prng.base_model.model.lin2.randlora_gamma["other"].data_ptr()
        )

    def test_randlora_different_shapes(self, mlp):
        config = RandLoraConfig(target_modules=["lin0", "lin3"], init_weights=False)
        mlp_different_shapes = get_peft_model(mlp, config)

        randlora_A = mlp_different_shapes.randlora_A["default"]
        randlora_B = mlp_different_shapes.randlora_B["default"]

        # sanity check
        assert mlp.lin0.base_layer.weight.shape != mlp.lin3.base_layer.weight.shape

        # lin0 has the largest output dimension, lin3 has the largest input dimension
        # randlora_A should have the shape of (rank, largest_in), randlora_B should have the shape of (largest_out, rank)
        assert randlora_A.shape == (config.r, 1, mlp.lin3.in_features)
        assert randlora_B.shape == (mlp.lin0.out_features, 1, config.r)

        # should not raise
        input = torch.randn(5, 10)
        mlp_different_shapes(input)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
    def test_randlora_dtypes(self, dtype):
        if dtype == torch.bfloat16:
            # skip if bf16 is not supported on hardware, see #1872
            if not is_bf16_available():
                pytest.skip("bfloat16 not supported on this system, skipping the test")

        model = MLP().to(dtype)
        config = RandLoraConfig(target_modules=["lin1", "lin2"], init_weights=False)
        peft_model = get_peft_model(model, config)
        inputs = torch.randn(5, 10).to(dtype)
        output = peft_model(inputs)  # should not raise
        assert output.dtype == dtype

    @pytest.mark.parametrize("layer_type", ["linear", "conv1d"])
    @pytest.mark.parametrize("ranks", [(8, 4), (8, 3), (4, 8)])
    @pytest.mark.parametrize("sparse", [None, "sparse", "very_sparse"])
    @pytest.mark.parametrize("safe_merge", [False, True])
    def test_mixed_rank_adapters_forward_merge(
        self, ranks: tuple[int, int], sparse: str | None, safe_merge: bool, layer_type: str
    ) -> None:
        # Regression for #4125: adding an adapter must not change the first adapter's basis count.
        torch.manual_seed(0)
        if layer_type == "linear":
            base_model = nn.Sequential(nn.Linear(16, 16), nn.ReLU(), nn.Linear(16, 8))
        else:
            base_model = nn.Sequential(Conv1D(16, 16), nn.ReLU(), Conv1D(8, 16))
        model = get_peft_model(
            base_model,
            RandLoraConfig(
                r=ranks[0],
                target_modules=["0", "2"],
                init_weights=False,
                sparse=sparse == "sparse",
                very_sparse=sparse == "very_sparse",
                randlora_alpha=ranks[0],
                fan_in_fan_out=layer_type == "conv1d",
            ),
            adapter_name="first",
        ).eval()
        inputs = torch.randn(2, 16)
        expected_first = model(inputs).detach()
        model.add_adapter(
            "second",
            RandLoraConfig(
                r=ranks[1],
                target_modules=["0", "2"],
                init_weights=False,
                sparse=sparse == "sparse",
                very_sparse=sparse == "very_sparse",
                randlora_alpha=ranks[1],
                fan_in_fan_out=layer_type == "conv1d",
            ),
        )
        torch.testing.assert_close(model(inputs), expected_first)
        for adapter in ("first", "second"):
            model.set_adapter(adapter)
            expected = model(inputs)
            with model.disable_adapter():
                assert not torch.allclose(expected, model(inputs))
            expected.sum().backward()
            for layer in (base_model[0], base_model[2]):
                assert layer.randlora_lambda[adapter].grad is not None
                assert layer.randlora_gamma[adapter].grad is not None
            model.merge_adapter(safe_merge=safe_merge)
            torch.testing.assert_close(model(inputs), expected, atol=1e-5, rtol=1e-5)
            model.unmerge_adapter()
            torch.testing.assert_close(model(inputs), expected, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("save_projection", [False, True])
    @pytest.mark.parametrize("load_order", [("high", "low"), ("low", "high")])
    @pytest.mark.parametrize("sparse", [None, "sparse", "very_sparse"])
    def test_mixed_rank_adapters_roundtrip(
        self, tmp_path: Path, save_projection: bool, load_order: tuple[str, str], sparse: str | None
    ) -> None:
        torch.manual_seed(0)
        base_model = nn.Sequential(nn.Linear(16, 16), nn.Linear(16, 8))
        original = copy.deepcopy(base_model)
        model = get_peft_model(
            base_model,
            RandLoraConfig(
                r=8, target_modules=["0", "1"], init_weights=False, save_projection=save_projection, sparse=sparse
            ),
            adapter_name="high",
        ).eval()
        model.add_adapter(
            "low",
            RandLoraConfig(
                r=4, target_modules=["0", "1"], init_weights=False, save_projection=save_projection, sparse=sparse
            ),
        )
        inputs = torch.randn(2, 16)
        expected = {}
        for adapter in ("high", "low"):
            model.set_adapter(adapter)
            expected[adapter] = model(inputs).detach()
        model.save_pretrained(tmp_path)
        first, second = load_order
        loaded = PeftModel.from_pretrained(original, tmp_path / first, adapter_name=first).eval()
        loaded.load_adapter(tmp_path / second, adapter_name=second)
        for adapter in ("high", "low"):
            loaded.set_adapter(adapter)
            torch.testing.assert_close(loaded(inputs), expected[adapter])

    def test_update_layer_slices_oversized_projection_rank(self) -> None:
        # Shared projections are produced for the largest layer. A smaller layer can use their rank prefix.
        torch.manual_seed(0)
        model = get_peft_model(
            nn.Sequential(nn.Linear(16, 16), nn.Linear(16, 8)),
            RandLoraConfig(r=8, target_modules=["0", "1"], init_weights=False, randlora_alpha=8),
        )
        layer = model.base_model.model[1]
        inputs = torch.randn(2, 16)
        expected_default = layer(inputs).detach()
        config = RandLoraConfig(r=4, target_modules=["1"], init_weights=False, randlora_alpha=4)
        layer.update_layer("low", model.randlora_A, model.randlora_B, r=4, config=config)
        torch.testing.assert_close(layer(inputs), expected_default)
        layer.set_adapter("low")
        expected = layer(inputs)
        expected.sum().backward()
        assert layer.randlora_lambda["low"].grad is not None
        layer.merge()
        torch.testing.assert_close(layer(inputs), expected)

    def test_same_rank_adapters_share_projections_after_mixed_ranks(self) -> None:
        model = get_peft_model(nn.Sequential(nn.Linear(16, 16)), RandLoraConfig(r=8, target_modules=["0"]))
        for name in ("low", "other_low"):
            model.add_adapter(name, RandLoraConfig(r=4, target_modules=["0"]))
        assert model.randlora_A["low"] is model.randlora_A["other_low"]
        assert model.randlora_B["low"] is model.randlora_B["other_low"]

    def test_mixed_rank_weighted_adapter_is_not_supported(self) -> None:
        model = get_peft_model(
            nn.Sequential(nn.Linear(16, 16)), RandLoraConfig(r=8, target_modules=["0"], init_weights=False)
        )
        model.add_adapter("low", RandLoraConfig(r=4, target_modules=["0"], init_weights=False))
        inputs = torch.randn(2, 16)
        expected = model(inputs).detach()
        # RandLoRA has no weighted-adapter API. Do not accidentally invoke another tuner's implementation.
        with pytest.raises(AttributeError, match="add_weighted_adapter"):
            model.add_weighted_adapter(["default", "low"], [0.5, 0.5], "combined")
        torch.testing.assert_close(model(inputs), expected)

    @pytest.mark.parametrize("deleted_adapter", ["default", "other"])
    def test_delete_adapter_cleans_basis_count(self, deleted_adapter: str) -> None:
        model = get_peft_model(
            nn.Sequential(nn.Linear(16, 16)), RandLoraConfig(r=8, target_modules=["0"], init_weights=False)
        )
        model.add_adapter("other", RandLoraConfig(r=4, target_modules=["0"], init_weights=False))
        remaining = "other" if deleted_adapter == "default" else "default"
        model.set_adapter(remaining)
        inputs = torch.randn(2, 16)
        expected = model(inputs).detach()
        model.delete_adapter(deleted_adapter)
        layer = model.base_model.model[0]
        assert layer.num_bases == {remaining: 4 if remaining == "other" else 2}
        for name in layer.adapter_layer_names + layer.other_param_names:
            assert deleted_adapter not in getattr(layer, name)
        torch.testing.assert_close(model(inputs), expected)
        model.add_adapter(deleted_adapter, RandLoraConfig(r=3, target_modules=["0"], init_weights=False))
        model.set_adapter(deleted_adapter)
        output = model(inputs)
        model.merge_adapter()
        torch.testing.assert_close(model(inputs), output, atol=1e-4, rtol=1e-4)
