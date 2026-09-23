"""Run the exported first-chunk and stateful recurrent OpenVINO DiT graphs."""

from pathlib import Path
import time

import numpy as np
import openvino as ov
import torch


class StatefulDiT:
    def __init__(self, ir_dir):
        self.ir_dir = Path(ir_dir).resolve()
        for filename in ("dit_forward_first.xml", "dit_forward_next.xml"):
            if not (self.ir_dir / filename).is_file():
                raise FileNotFoundError(self.ir_dir / filename)
        self.core = ov.Core()
        self.first = None
        self.next = None
        self.request = None
        self.state_names = None
        self.expected_idx = 0

    @staticmethod
    def _array(tensor):
        return np.ascontiguousarray(tensor.detach().to(device="cpu", dtype=torch.float32).numpy())

    def reset(self):
        if self.request is not None:
            self.request.reset_state()
        self.expected_idx = 0

    def _load_first(self):
        if self.first is None:
            start = time.monotonic()
            self.first = self.core.compile_model(str(self.ir_dir / "dit_forward_first.xml"), "CPU")
            print(f"[stateful DiT] compiled first IR in {time.monotonic() - start:.1f}s", flush=True)

    def _load_next(self):
        if self.next is not None:
            return
        start = time.monotonic()
        model = self.core.read_model(str(self.ir_dir / "dit_forward_next.xml"))
        self.state_names = {
            op.get_variable_id(): next(iter(op.output(0).get_names()))
            for op in model.get_ops() if op.get_type_name() == "ReadValue"
        }
        expected = {f"cache_{kind}_{i:02d}" for kind in ("k", "v") for i in range(30)}
        if set(self.state_names.values()) != expected:
            raise RuntimeError("Recurrent IR has unexpected KV variable names")
        self.next = self.core.compile_model(model, "CPU")
        self.request = self.next.create_infer_request()
        print(f"[stateful DiT] compiled recurrent IR in {time.monotonic() - start:.1f}s", flush=True)

    def __call__(self, x, timestep, lq_latents, process_idx):
        if process_idx != self.expected_idx:
            raise ValueError(f"Expected chunk {self.expected_idx}, got {process_idx}; reset before another video")
        if x.device.type != "cpu" or x.dtype != torch.float32:
            raise ValueError("Saved DiT IR requires CPU FP32 latents")
        if x.shape[-2] < 16 or x.shape[-1] < 16 or x.shape[-2] % 16 or x.shape[-1] % 16:
            raise ValueError("Stateful DiT requires frame height and width to be multiples of 128")
        if x.shape[2] != (6 if process_idx == 0 else 2):
            raise ValueError(f"Unexpected chunk length {x.shape[2]}")
        lq = self._array(torch.stack(lq_latents, dim=0))
        inputs = {"x": self._array(x), "timestep": self._array(timestep),
                  "lq_latents_stacked": lq}
        start = time.monotonic()
        if process_idx == 0:
            self._load_first()
            first_request = self.first.create_infer_request()
            first_request.infer(inputs)
            noise = first_request.get_output_tensor(0).data.copy()
            caches = {}
            for i in range(30):
                caches[f"cache_k_{i:02d}"] = first_request.get_output_tensor(1 + i).data.copy()
                caches[f"cache_v_{i:02d}"] = first_request.get_output_tensor(31 + i).data.copy()
            self._load_next()
            self.request.reset_state()
            states = self.request.query_state()
            if set(s.name for s in states) != set(self.state_names):
                raise RuntimeError("Compiled IR state IDs do not match the saved IR")
            for state in states:
                state.state = ov.Tensor(caches[self.state_names[state.name]])
            print(f"[stateful DiT] seeded {len(states)} recurrent K/V states", flush=True)
        else:
            inputs["process_idx"] = np.array([process_idx], dtype=np.int64)
            self.request.infer(inputs)
            noise = self.request.get_output_tensor(0).data.copy()
        self.expected_idx += 1
        print(f"[stateful DiT] chunk {process_idx} inferred in {time.monotonic() - start:.1f}s", flush=True)
        return torch.from_numpy(noise)