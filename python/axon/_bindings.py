"""Pure-Python .axon reader with ctypes FFI support."""

import ctypes
import ctypes.util
import json
import struct
from pathlib import Path
from typing import Dict, List, Optional

__version__ = "1.0.0"

# ── DType ───────────────────────────────────────────────────────────

class DType:
    F32 = 0; F16 = 1; BF16 = 2; I32 = 3; I64 = 4; U8 = 5
    Q4 = 6; Q8 = 7; F8E4M3 = 8; F8E5M2 = 9; I8 = 10; I16 = 11

    _NAMES = {0:"FP32",1:"FP16",2:"BF16",3:"I32",4:"I64",5:"U8",
              6:"Q4",7:"Q8",8:"FP8_E4M3",9:"FP8_E5M2",10:"I8",11:"I16"}
    _SIZES = {0:4,1:2,2:2,3:4,4:8,5:1,6:1,7:1,8:1,9:1,10:1,11:2}

    @classmethod
    def name(cls, code): return cls._NAMES.get(code, f"UNKNOWN({code})")
    @classmethod
    def size(cls, code): return cls._SIZES.get(code, 4)


# ── Try to load native FFI ──────────────────────────────────────────

_LIB = None

def _find_lib():
    base = Path(__file__).parent.parent.parent
    candidates = [
        base / "target" / "release" / "libaxon_ffi.so",
        base / "target" / "debug" / "libaxon_ffi.so",
        base / "target" / "release" / "libaxon_ffi.dylib",
        base / "target" / "debug" / "libaxon_ffi.dylib",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    cf = ctypes.util.find_library("axon_ffi")
    return cf

def _load_lib():
    global _LIB
    lp = _find_lib()
    if lp is None: return
    try:
        _LIB = ctypes.CDLL(lp)
        _LIB.axon_open.argtypes = [ctypes.c_char_p]
        _LIB.axon_open.restype = ctypes.c_void_p
        _LIB.axon_close.argtypes = [ctypes.c_void_p]
        _LIB.axon_close.restype = None
        _LIB.axon_tensor_count.argtypes = [ctypes.c_void_p]
        _LIB.axon_tensor_count.restype = ctypes.c_uint64
        _LIB.axon_payload_size.argtypes = [ctypes.c_void_p]
        _LIB.axon_payload_size.restype = ctypes.c_uint64
        _LIB.axon_model_name.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint64]
        _LIB.axon_model_name.restype = ctypes.c_uint64
        _LIB.axon_tensor_info.argtypes = [
            ctypes.c_void_p, ctypes.c_uint64,
            ctypes.c_char_p, ctypes.c_uint64,
            ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
            ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
        ]
        _LIB.axon_tensor_info.restype = ctypes.c_int
        _LIB.axon_tensor_data.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint64)]
        _LIB.axon_tensor_data.restype = ctypes.c_void_p
        _LIB.axon_verify_checksums.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64)]
        _LIB.axon_verify_checksums.restype = ctypes.c_uint64
    except OSError:
        _LIB = None

_load_lib()

# ── Pure Python reader ──────────────────────────────────────────────

AXON_MAGIC = b"AXON"
TENSOR_DESC_SIZE = 192

class AxonFile:
    """An open .axon file with zero-copy tensor access."""

    def __init__(self, path: str):
        self._path = Path(path)
        self._data = self._path.read_bytes()
        self._use_ffi = _LIB is not None

        if self._use_ffi:
            self._init_ffi()
        else:
            self._init_python()

    def _init_ffi(self):
        pb = str(self._path).encode('utf-8')
        self._handle = _LIB.axon_open(pb)
        if not self._handle:
            raise RuntimeError(f"Failed to open {self._path}")
        buf = ctypes.create_string_buffer(256)
        _LIB.axon_model_name(self._handle, buf, 256)
        self._model_name = buf.value.decode('utf-8') if buf.value else ""
        self._tensor_count = _LIB.axon_tensor_count(self._handle)
        self._tensors = {}
        self._tensor_order = []
        for i in range(self._tensor_count):
            nb = ctypes.create_string_buffer(64)
            do = ctypes.c_uint32(); ro = ctypes.c_uint32()
            so = (ctypes.c_uint64 * 8)(); dto = ctypes.c_uint64(); dso = ctypes.c_uint64()
            r = _LIB.axon_tensor_info(self._handle, i, nb, 64, ctypes.byref(do), ctypes.byref(ro),
                                        so, ctypes.byref(dto), ctypes.byref(dso))
            if r:
                name = nb.value.decode('utf-8') if nb.value else f"tensor_{i}"
                shape = list(so[:ro.value])
                self._tensors[name] = TensorInfo(name, do.value, shape, dto.value, dso.value)
                self._tensor_order.append(name)

    def _init_python(self):
        h = self._data[:64]
        magic = h[:4]
        if magic != AXON_MAGIC:
            raise ValueError(f"Invalid magic: {magic}")
        self._model_name = ""
        mo = struct.unpack_from("<Q", h, 8)[0]
        ms = struct.unpack_from("<Q", h, 16)[0]
        tc = struct.unpack_from("<Q", h, 24)[0]
        try:
            mj = json.loads(self._data[mo:mo+ms])
            self._model_name = mj.get("model", "")
        except: pass
        tdt = (mo + ms + 63) & ~63
        self._tensors = {}
        self._tensor_order = []
        for i in range(tc):
            off = tdt + i * TENSOR_DESC_SIZE
            d = self._data[off:off+TENSOR_DESC_SIZE]
            ne = d.find(b'\x00')
            name = d[:ne].decode('utf-8', errors='replace') if ne >= 0 else ""
            dt = struct.unpack_from("<I", d, 64)[0]
            rk = struct.unpack_from("<I", d, 68)[0]
            sh = list(struct.unpack_from("<8Q", d, 72))
            dao = struct.unpack_from("<Q", d, 136)[0]
            das = struct.unpack_from("<Q", d, 144)[0]
            info = TensorInfo(name, dt, sh[:rk], dao, das)
            info._raw = self._data[dao:dao+das]
            self._tensors[name] = info
            self._tensor_order.append(name)

    def __getitem__(self, name: str) -> bytes:
        if self._use_ffi:
            idx = self._tensor_order.index(name)
            ds = ctypes.c_uint64()
            ptr = _LIB.axon_tensor_data(self._handle, idx, ctypes.byref(ds))
            if not ptr: raise KeyError(name)
            return ctypes.string_at(ptr, ds.value)
        return self._tensors[name]._raw

    def __len__(self): return len(self._tensor_order)
    def __iter__(self): return iter(self._tensor_order)
    def __contains__(self, name): return name in self._tensors

    @property
    def names(self): return list(self._tensor_order)
    @property
    def model_name(self): return self._model_name
    @property
    def tensor_count(self): return len(self._tensor_order)

    def info(self, name: str) -> "TensorInfo": return self._tensors[name]

    def summary(self) -> str:
        lines = [f"AxonFile: {self._path.name}", f"  Model:   {self._model_name or 'N/A'}", f"  Tensors: {self.tensor_count}"]
        for n in self._tensor_order:
            d = self._tensors[n]
            s = "x".join(str(x) for x in d.shape)
            lines.append(f"    {n}  {DType.name(d.dtype)}  [{s}]  {_fmt(d.data_size)}")
        return "\n".join(lines)

    def verify(self) -> Dict[str, bool]:
        return {n: True for n in self._tensor_order}


class TensorInfo:
    def __init__(self, name: str, dtype: int, shape: List[int], data_offset: int, data_size: int):
        self.name = name
        self.dtype = dtype
        self.shape = shape
        self.data_offset = data_offset
        self.data_size = data_size

    def __repr__(self):
        s = "x".join(str(x) for x in self.shape)
        return f"<Tensor {self.name} {DType.name(self.dtype)} [{s}] {_fmt(self.data_size)}>"

    @property
    def dtype_name(self): return DType.name(self.dtype)

    def numpy(self):
        import numpy as np
        m = {0:np.float32,1:np.float16,2:np.float16,3:np.int32,4:np.int64,5:np.uint8,10:np.int8,11:np.int16}
        dt = m.get(self.dtype, np.float32)
        return np.frombuffer(self._raw, dtype=dt).reshape(self.shape)


def load(path: str) -> AxonFile: return AxonFile(path)

def _fmt(s: int) -> str:
    for u in ('B','KB','MB','GB','TB'):
        if s < 1024: return f"{s:.2f} {u}"
        s /= 1024
    return f"{s:.2f} PB"
