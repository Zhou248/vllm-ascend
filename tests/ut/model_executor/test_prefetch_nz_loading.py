from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from vllm.model_executor.model_loader import utils as loader_utils
from vllm.model_executor.offloader.base import NoopOffloader

from vllm_ascend.model_executor.offloader import loading
from vllm_ascend.model_executor.offloader.prefetch import _ModuleOffloader


def test_raw_view_checks_storage_bounds():
    host = torch.arange(16)
    view = loading._raw_host_view(host, (2, 3), (4, 1))
    assert view.tolist() == [[0, 1, 2], [4, 5, 6]]
    with pytest.raises(ValueError, match="exceeds"):
        loading._raw_host_view(host, (5, 5), (5, 1))
    with pytest.raises(ValueError, match="Negative"):
        loading._raw_host_view(host, (2,), (-1,))


def test_release_only_selected_mxfp8_references():
    selected, resident = torch.ones(2), torch.zeros(2)
    module = SimpleNamespace(_mxfp8_moe_buffers={"w13": (selected,), "w2": (resident,)})
    loading._release_offloaded_mxfp8_buffers(module, {selected.data_ptr()})
    assert list(module._mxfp8_moe_buffers) == ["w2"]


def test_parameter_replacement_keeps_raw_host_and_updates_owner(monkeypatch):
    module = torch.nn.Linear(3, 2, bias=False)
    old_storage = module.weight.data

    class Owner:
        _cpu_storage = old_storage

        @property
        def _param(self):
            return module.weight

    owner = Owner()
    layer = SimpleNamespace(_param_offloaders={"weight": owner}, _nz_formats={}, _nz_host_storages={})
    offloader = SimpleNamespace(module_offloaders=[layer])
    host = torch.arange(6, dtype=torch.float32)
    monkeypatch.setattr(torch.Tensor, "to", lambda self, *args, **kwargs: self)
    monkeypatch.setattr(loading, "_get_nz_format", lambda tensor: 29)
    monkeypatch.setattr(loading, "_raw_memcpy_available", lambda: True)
    monkeypatch.setattr(loading, "_capture_nz_host_storage", lambda tensor: host)
    with loading._prefetch_loading_context(module, SimpleNamespace(type="npu"), offloader):
        module.weight = torch.nn.Parameter(torch.ones(2, 3))
        module._mxfp8_moe_buffers = {"weight": (module.weight.data,)}
    assert module.weight._ascend_prefetch_nz_host is host
    assert owner._cpu_storage.data_ptr() == host.data_ptr()
    assert module._mxfp8_moe_buffers == {}
    infos = _ModuleOffloader.get_param_infos(layer)
    assert infos[0].nz_format == 29
    assert layer._nz_host_storages["weight"] is host


def test_missing_raw_operator_fails_without_silent_copy(monkeypatch):
    module = torch.nn.Linear(2, 2, bias=False)
    owner = SimpleNamespace(_param=module.weight)
    offloader = SimpleNamespace(module_offloaders=[SimpleNamespace(_param_offloaders={"weight": owner})])
    monkeypatch.setattr(torch.Tensor, "to", lambda self, *args, **kwargs: self)
    monkeypatch.setattr(loading, "_get_nz_format", lambda tensor: 29)
    monkeypatch.setattr(loading, "_raw_memcpy_available", lambda: False)
    with (
        pytest.raises(RuntimeError, match="swap_blocks_batch"),
        loading._prefetch_loading_context(module, SimpleNamespace(type="npu"), offloader),
    ):
        pass


def test_unowned_nz_parameter_is_rejected(monkeypatch):
    module = torch.nn.Linear(2, 2, bias=False)
    monkeypatch.setattr(torch.Tensor, "to", lambda self, *args, **kwargs: self)
    monkeypatch.setattr(loading, "_get_nz_format", lambda tensor: 29)
    with (
        pytest.raises(RuntimeError, match="not owned"),
        loading._prefetch_loading_context(module, SimpleNamespace(type="npu"), SimpleNamespace(module_offloaders=[])),
    ):
        pass


def test_non_prefetch_loading_keeps_upstream_context(monkeypatch):
    calls = []

    @contextmanager
    def original(module, device):
        calls.append((module, device))
        yield module

    monkeypatch.setattr(loader_utils, "device_loading_context", original)
    monkeypatch.setattr(loading, "get_offloader", lambda: NoopOffloader())
    loading.install_prefetch_loading_context()
    installed = loader_utils.device_loading_context
    loading.install_prefetch_loading_context()
    assert loader_utils.device_loading_context is installed
    module, device = object(), SimpleNamespace(type="npu")
    with installed(module, device) as value:
        assert value is module
    assert calls == [(module, device)]
