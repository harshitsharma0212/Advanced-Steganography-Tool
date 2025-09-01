#!/usr/bin/env python3
"""
Stego Universal — Fixed & Improved
- Secret: file or text (text limit: 10,000 words)
- Cover: PNG/BMP/JPG (image LSB), WAV (audio LSB), MP4/MOV/AVI (video LSB via OpenCV), or any file (append-mode)
- Crypto: AES-256-GCM (PBKDF2)
- GUI: CustomTkinter with tabs, preview, logs, animations
Author: ChatGPT (tailored)
"""
import os
import struct
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox

# Crypto
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

# Imaging and arrays
from PIL import Image, ImageTk
import numpy as np

# Audio
import soundfile as sf

# GUI
import customtkinter as ctk

# Optional libs
try:
    import cv2
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False

# Output folder for extracted files
OUTPUT_DIR = "extracted_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------- Crypto helpers ----------------
KDF_ITERS = 300_000


def kdf_derive(password: str, salt: bytes) -> bytes:
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=KDF_ITERS,
                     backend=default_backend())
    return kdf.derive(password.encode())


def encrypt_bytes(password: str, plaintext: bytes) -> bytes:
    salt = os.urandom(16)
    key = kdf_derive(password, salt)
    aes = AESGCM(key)
    nonce = os.urandom(12)
    ct = aes.encrypt(nonce, plaintext, None)
    # Format: "SGU1" | salt(16) | nonce(12) | ciphertext
    return b"SGU1" + salt + nonce + ct


def decrypt_bytes(password: str, blob: bytes):
    try:
        if not blob.startswith(b"SGU1"):
            return None
        salt = blob[4:20]
        nonce = blob[20:32]
        ct = blob[32:]
        key = kdf_derive(password, salt)
        aes = AESGCM(key)
        return aes.decrypt(nonce, ct, None)
    except Exception:
        return None


# ---------------- Payload pack/unpack ----------------
MAGIC = b"STGU"  # primary marker


def pack_payload(ptype: int, name: str, enc_blob: bytes) -> bytes:
    """
    ptype: 0=text, 1=file
    format: MAGIC + ptype(1) + name_len(2) + data_len(4) + name + enc_blob
    """
    name_b = (name or "").encode('utf-8')
    header = MAGIC + struct.pack(">BHI", ptype, len(name_b), len(enc_blob)) + name_b
    return header + enc_blob


def unpack_payload(buf: bytes):
    try:
        if not buf or not buf.startswith(MAGIC):
            return None
        ptype = buf[4]
        name_len = struct.unpack(">H", buf[5:7])[0]
        data_len = struct.unpack(">I", buf[7:11])[0]
        name = buf[11:11 + name_len].decode('utf-8', errors='ignore')
        payload = buf[11 + name_len:11 + name_len + data_len]
        return ptype, name, payload
    except Exception:
        return None


# ---------------- wrappers for single / split transport ----------------
SINGLE_TAG = b"STGU-ONE"
SPLIT_TAG = b"STGU-PART"


def pack_single_part(payload: bytes) -> bytes:
    return SINGLE_TAG + struct.pack(">I", len(payload)) + payload


def unwrap_single_part(buf: bytes) -> bytes:
    if not buf.startswith(SINGLE_TAG):
        raise ValueError("Not a single-part payload")
    L = struct.unpack(">I", buf[len(SINGLE_TAG):len(SINGLE_TAG) + 4])[0]
    return buf[len(SINGLE_TAG) + 4:len(SINGLE_TAG) + 4 + L]


def wrap_part(part_idx: int, total_parts: int, chunk: bytes) -> bytes:
    header = SPLIT_TAG + struct.pack(">HHI", part_idx, total_parts, len(chunk))
    return header + chunk


def unwrap_part(buf: bytes):
    if not buf.startswith(SPLIT_TAG):
        return None
    part_idx, total_parts, clen = struct.unpack(">HHI", buf[len(SPLIT_TAG):len(SPLIT_TAG) + 8])
    chunk = buf[len(SPLIT_TAG) + 8:len(SPLIT_TAG) + 8 + clen]
    return part_idx, total_parts, chunk


def split_payload_into_chunks(packed: bytes, capacities: list):
    chunks = []
    offset = 0
    total = len(packed)
    for cap in capacities:
        if offset >= total:
            chunks.append(b"")
            continue
        take = min(cap, total - offset)
        part = packed[offset:offset + take]
        chunks.append(part)
        offset += take
    if offset < total:
        raise ValueError(f"Total capacity across selected images is insufficient. Need {total - offset} more bytes.")
    return chunks


# ---------------- bit helpers ----------------
def bytes_to_bits(data: bytes) -> str:
    return ''.join(f'{b:08b}' for b in data)


def bits_to_bytes(bits: str) -> bytes:
    b = bytearray()
    for i in range(0, len(bits), 8):
        chunk = bits[i:i + 8]
        if len(chunk) < 8:
            break
        b.append(int(chunk, 2))
    return bytes(b)


# ---------------- Image helpers ----------------
def img_capacity_bytes(img: Image.Image, bits_per_channel: int = 1) -> int:
    channels = len(img.getbands())
    total_bits = img.width * img.height * channels * bits_per_channel
    return total_bits // 8


def embed_in_image(cover_path: str, payload: bytes, out_path: str, bits_per_channel: int = 1, progress_callback=None):
    img = Image.open(cover_path)
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGBA")
    arr = np.array(img, dtype=np.uint8)
    flat = arr.flatten()
    total_bits = flat.size * bits_per_channel
    need_bits = len(payload) * 8
    if need_bits > total_bits:
        raise ValueError(f"Payload {len(payload)} bytes > image capacity {total_bits // 8} bytes.")
    bits = bytes_to_bits(payload)
    mask_clear = ~((1 << bits_per_channel) - 1) & 0xFF
    bit_idx = 0
    step = max(1, flat.size // 100)
    for i in range(flat.size):
        if bit_idx >= len(bits):
            break
        chunk = bits[bit_idx:bit_idx + bits_per_channel]
        if len(chunk) < bits_per_channel:
            chunk = chunk.ljust(bits_per_channel, '0')
        val = int(chunk, 2)
        flat[i] = (int(flat[i]) & mask_clear) | val
        bit_idx += bits_per_channel
        if progress_callback and (i % step == 0):
            progress_callback(int(i / flat.size * 100))
    if progress_callback:
        progress_callback(100)
    new_arr = flat.reshape(arr.shape)
    out_img = Image.fromarray(new_arr.astype(np.uint8), mode=img.mode)
    out_img.save(out_path)


def extract_from_image(stego_path: str, bits_per_channel: int = 1, progress_callback=None):
    img = Image.open(stego_path)
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGBA")
    arr = np.array(img, dtype=np.uint8)
    flat = arr.flatten()
    bitstr = []
    bit_count_needed = None
    step = max(1, flat.size // 100)
    for i in range(flat.size):
        lsb_val = int(flat[i]) & ((1 << bits_per_channel) - 1)
        chunk = format(lsb_val, f'0{bits_per_channel}b')
        bitstr.append(chunk)
        current_bits = ''.join(bitstr)
        # header detection: need at least enough bits for MAGIC + 1 + 2 + 4 = 11 bytes
        if len(current_bits) >= 8 * 11 and bit_count_needed is None:
            header_bytes = bits_to_bytes(current_bits[:8 * 11])
            if header_bytes.startswith(MAGIC):
                try:
                    name_len = struct.unpack(">H", header_bytes[5:7])[0]
                    data_len = struct.unpack(">I", header_bytes[7:11])[0]
                    total_payload_bytes = 11 + name_len + data_len
                    bit_count_needed = total_payload_bytes * 8
                except Exception:
                    pass
        if bit_count_needed is not None and len(current_bits) >= bit_count_needed:
            if progress_callback:
                progress_callback(100)
            return bits_to_bytes(current_bits[:bit_count_needed])
        if progress_callback and (i % step == 0):
            progress_callback(int(i / flat.size * 100))
    if progress_callback:
        progress_callback(100)
    current_bits = ''.join(bitstr)
    data = bits_to_bytes(current_bits)
    if data.startswith(MAGIC):
        return data
    return None


# ---------------- WAV helpers ----------------
def wav_capacity_bytes(wav_path: str, bits_per_sample: int = 1) -> int:
    data, _ = sf.read(wav_path, dtype='int16')
    total_samples = data.size
    return (total_samples * bits_per_sample) // 8


def embed_in_wav(cover_wav: str, payload: bytes, out_path: str, bits_per_sample: int = 1, progress_callback=None):
    data, sr = sf.read(cover_wav, dtype='int16')
    flat = data.flatten().astype(np.int16)
    cap = (flat.size * bits_per_sample) // 8
    if len(payload) > cap:
        raise ValueError(f"Payload {len(payload)} bytes > wav capacity {cap} bytes.")
    bits = bytes_to_bits(payload)
    mask_clear = ~((1 << bits_per_sample) - 1) & 0xFFFF
    bit_idx = 0
    step = max(1, flat.size // 100)
    for i in range(flat.size):
        if bit_idx >= len(bits):
            break
        chunk = bits[bit_idx:bit_idx + bits_per_sample]
        if len(chunk) < bits_per_sample:
            chunk = chunk.ljust(bits_per_sample, '0')
        val = int(chunk, 2)
        flat[i] = (int(flat[i]) & mask_clear) | val
        bit_idx += bits_per_sample
        if progress_callback and (i % step == 0):
            progress_callback(int(i / flat.size * 100))
    if progress_callback:
        progress_callback(100)
    new = flat.reshape(data.shape)
    sf.write(out_path, new, sr, subtype='PCM_16')


def extract_from_wav(stego_wav: str, bits_per_sample: int = 1, progress_callback=None):
    data, _ = sf.read(stego_wav, dtype='int16')
    flat = data.flatten()
    bitstr = []
    bit_count_needed = None
    step = max(1, flat.size // 100)
    for i in range(flat.size):
        lsb_val = int(flat[i]) & ((1 << bits_per_sample) - 1)
        chunk = format(lsb_val, f'0{bits_per_sample}b')
        bitstr.append(chunk)
        current_bits = ''.join(bitstr)
        if len(current_bits) >= 8 * 11 and bit_count_needed is None:
            header_bytes = bits_to_bytes(current_bits[:8 * 11])
            if header_bytes.startswith(MAGIC):
                try:
                    name_len = struct.unpack(">H", header_bytes[5:7])[0]
                    data_len = struct.unpack(">I", header_bytes[7:11])[0]
                    total_payload_bytes = 11 + name_len + data_len
                    bit_count_needed = total_payload_bytes * 8
                except Exception:
                    pass
        if bit_count_needed is not None and len(current_bits) >= bit_count_needed:
            if progress_callback:
                progress_callback(100)
            return bits_to_bytes(current_bits[:bit_count_needed])
        if progress_callback and (i % step == 0):
            progress_callback(int(i / flat.size * 100))
    if progress_callback:
        progress_callback(100)
    current_bits = ''.join(bitstr)
    data = bits_to_bytes(current_bits)
    if data.startswith(MAGIC):
        return data
    return None


# ---------------- Video helpers (same idea) ----------------
def video_capacity_bytes(video_path: str, bits_per_channel: int = 1) -> int:
    if not HAS_CV2:
        raise RuntimeError('OpenCV (cv2) required for video support')
    cap = cv2.VideoCapture(video_path)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    channels = 3
    total_bits = frames * w * h * channels * bits_per_channel
    cap.release()
    return total_bits // 8


def embed_in_video(cover_video: str, payload: bytes, out_path: str, bits_per_channel: int = 1, progress_callback=None):
    if not HAS_CV2:
        raise RuntimeError('OpenCV (cv2) required for video support')
    cap = cv2.VideoCapture(cover_video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_bits = frames * w * h * 3 * bits_per_channel
    need_bits = len(payload) * 8
    if need_bits > total_bits:
        cap.release()
        raise ValueError('Payload too large for video capacity')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    bits = bytes_to_bits(payload)
    bit_idx = 0
    frame_idx = 0
    step = max(1, frames // 100)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        flat = frame.flatten()
        for i in range(flat.size):
            if bit_idx >= len(bits):
                break
            chunk = bits[bit_idx:bit_idx + bits_per_channel]
            if len(chunk) < bits_per_channel:
                chunk = chunk.ljust(bits_per_channel, '0')
            val = int(chunk, 2)
            flat[i] = (int(flat[i]) & (~((1 << bits_per_channel) - 1) & 0xFF)) | val
            bit_idx += bits_per_channel
        new_frame = flat.reshape(frame.shape).astype(np.uint8)
        out.write(new_frame)
        frame_idx += 1
        if progress_callback and (frame_idx % step == 0):
            progress_callback(int(frame_idx / frames * 100))
    # write remaining frames (if any)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
    cap.release()
    out.release()
    if progress_callback:
        progress_callback(100)


def extract_from_video(stego_video: str, bits_per_channel: int = 1, progress_callback=None):
    if not HAS_CV2:
        raise RuntimeError('OpenCV (cv2) required for video support')
    cap = cv2.VideoCapture(stego_video)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    bits_collected = []
    step = max(1, frames // 100)
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        flat = frame.flatten()
        for v in flat:
            lsb_val = int(v) & ((1 << bits_per_channel) - 1)
            bits_collected.append(format(lsb_val, f'0{bits_per_channel}b'))
            current_bits = ''.join(bits_collected)
            if len(current_bits) >= 8 * 11:
                header_bytes = bits_to_bytes(current_bits[:8 * 11])
                if header_bytes.startswith(MAGIC):
                    try:
                        name_len = struct.unpack(">H", header_bytes[5:7])[0]
                        data_len = struct.unpack(">I", header_bytes[7:11])[0]
                        total_payload_bytes = 11 + name_len + data_len
                        need_bits = total_payload_bytes * 8
                        if len(current_bits) >= need_bits:
                            cap.release()
                            if progress_callback:
                                progress_callback(100)
                            return bits_to_bytes(current_bits[:need_bits])
                    except Exception:
                        pass
        frame_idx += 1
        if progress_callback and (frame_idx % step == 0):
            progress_callback(int(frame_idx / frames * 100))
    cap.release()
    if progress_callback:
        progress_callback(100)
    current_bits = ''.join(bits_collected)
    data = bits_to_bytes(current_bits)
    if data.startswith(MAGIC):
        return data
    return None


# ---------------- Append fallback ----------------
APPEND_MAGIC = b"STG-AF1"


def embed_by_append(cover_path: str, payload: bytes, out_path: str, progress_callback=None):
    with open(cover_path, 'rb') as f:
        base = f.read()
    with open(out_path, 'wb') as f:
        f.write(base)
        f.write(APPEND_MAGIC)
        f.write(struct.pack(">I", len(payload)))
        f.write(payload)
    if progress_callback:
        progress_callback(100)


def extract_by_append(stego_path: str):
    with open(stego_path, 'rb') as f:
        data = f.read()
    idx = data.rfind(APPEND_MAGIC)
    if idx == -1:
        return None
    pos = idx + len(APPEND_MAGIC)
    payload_len = struct.unpack(">I", data[pos:pos + 4])[0]
    start = pos + 4
    return data[start:start + payload_len]


# ---------------- GUI Application ----------------
class StegoUniversalApp:
    WORD_LIMIT = 10_000  # 10k words limit

    def __init__(self, root):
        self.root = root
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")
        root.title("Stego Universal — Fixed")
        root.geometry("1280x860")
        root.minsize(1150, 760)

        # fonts
        self.h1 = ctk.CTkFont(size=26, weight="bold")
        self.h2 = ctk.CTkFont(size=18, weight="bold")
        self.bf = ctk.CTkFont(size=15)
        self.mono = ctk.CTkFont(size=13, family="Consolas")

        # state
        self.cover_path = None
        self.secret_path = None
        self.secret_text = None
        self.stego_path = None
        self.stego_paths_multi = None

        self._build_ui()

    def _build_ui(self):
        header = ctk.CTkFrame(self.root)
        header.pack(fill="x", padx=12, pady=(10, 6))
        ctk.CTkLabel(header, text="Stego Universal — Fixed", font=self.h1).pack(anchor="w", padx=8)
        ctk.CTkLabel(header, text="Hide any file or text — AES-256-GCM encryption. Extract saves files to ./extracted_outputs",
                      font=self.bf, text_color="#cbd5e1").pack(anchor="w", padx=8, pady=(2, 8))

        self.tabs = ctk.CTkTabview(self.root, width=1240, height=650)
        self.tabs.pack(padx=12, pady=6, fill="both", expand=True)
        self.tab_hide = self.tabs.add("Hide")
        self.tab_extract = self.tabs.add("Extract")

        self._build_hide_tab()
        self._build_extract_tab()

        status = ctk.CTkFrame(self.root, height=40)
        status.pack(fill="x", padx=12, pady=(4, 12))
        self.status_label = ctk.CTkLabel(status, text="Status: Idle", font=self.bf)
        self.status_label.pack(side="left", padx=8)
        self.progress = ctk.CTkProgressBar(status, width=400)
        self.progress.pack(side="right", padx=8)

    # -- Hide Tab --
    def _build_hide_tab(self):
        frame = self.tab_hide
        left = ctk.CTkFrame(frame, width=380)
        left.pack(side="left", fill="y", padx=(12, 8), pady=8)
        left.pack_propagate(False)

        ctk.CTkLabel(left, text="Cover (Image/WAV/Video/Any)", font=self.h2).pack(anchor="w", padx=8, pady=(8, 6))
        ctk.CTkButton(left, text="Choose Cover", command=self.choose_cover, width=320, font=self.bf).pack(padx=8, pady=6)
        self.cover_info = ctk.CTkLabel(left, text="No cover selected", font=self.bf, wraplength=340, justify="left")
        self.cover_info.pack(padx=8, pady=(4, 8))

        ctk.CTkLabel(left, text="Secret (File) OR Text", font=self.h2).pack(anchor="w", padx=8, pady=(6, 4))
        btn_row = ctk.CTkFrame(left)
        btn_row.pack(padx=8, pady=(4, 4))
        ctk.CTkButton(btn_row, text="Choose Secret File", command=self.choose_secret_file, font=self.bf).pack(side="left", padx=6)
        ctk.CTkButton(btn_row, text="Open Text Editor", command=self.open_text_editor, font=self.bf).pack(side="left", padx=6)
        self.secret_info = ctk.CTkLabel(left, text="No secret selected", font=self.bf, wraplength=340, justify="left")
        self.secret_info.pack(padx=8, pady=(6, 12))

        ctk.CTkLabel(left, text="Password (AES-256 GCM)", font=self.h2).pack(anchor="w", padx=8, pady=(6, 4))
        self.pass_entry = ctk.CTkEntry(left, placeholder_text="Enter strong password", show="*", width=320, font=self.bf)
        self.pass_entry.pack(padx=8, pady=6)
        self.pass_strength = ctk.CTkLabel(left, text="Strength: —", font=self.bf)
        self.pass_strength.pack(anchor="w", padx=8)
        self.pass_entry.bind('<KeyRelease>', self._on_pass_change)

        ctk.CTkLabel(left, text="LSB / Stealth", font=self.h2).pack(anchor="w", padx=8, pady=(8, 4))
        self.bits_option = ctk.CTkOptionMenu(left, values=["1", "2", "3", "4"], command=self.update_capacity_info)
        self.bits_option.set("1")
        self.bits_option.pack(padx=8, pady=(0, 8))

        self.multi_split_var = tk.BooleanVar(value=False)
        self.multi_split_chk = ctk.CTkCheckBox(left, text="Allow multi-image split (image-only)", variable=self.multi_split_var)
        self.multi_split_chk.pack(anchor="w", padx=8, pady=(0, 8))

        self.capacity_label = ctk.CTkLabel(left, text="Capacity: --", font=self.bf, wraplength=340, justify="left")
        self.capacity_label.pack(padx=8, pady=(12, 8))

        ctk.CTkButton(left, text="Embed → Create Stego", command=self.embed_action, font=self.h2, width=320).pack(padx=8, pady=(10, 8))

        # center area
        center = ctk.CTkFrame(frame)
        center.pack(side="left", fill="both", expand=True, padx=(0, 8), pady=8)

        ctk.CTkLabel(center, text="Preview / Editor", font=self.h2).pack(anchor="w", padx=8, pady=(8, 6))
        self.preview_frame = ctk.CTkFrame(center, height=360)
        self.preview_frame.pack(fill="x", padx=8, pady=(0, 8))
        self.preview_frame.pack_propagate(False)
        self.preview_label = ctk.CTkLabel(self.preview_frame, text="No preview", font=self.bf)
        self.preview_label.pack(expand=True, fill="both", padx=8, pady=8)

        # text editor
        self.text_frame = ctk.CTkFrame(center)
        self.text_frame.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.text_frame.pack_forget()
        ctk.CTkLabel(self.text_frame, text="Secret Text Editor (max 10,000 words)", font=self.h2).pack(anchor="w", padx=6, pady=(6, 4))
        self.secret_textbox = ctk.CTkTextbox(self.text_frame, height=220, font=self.bf, wrap="word")
        self.secret_textbox.pack(fill="both", expand=True, padx=6, pady=(0, 6))
        self.text_bytes_label = ctk.CTkLabel(self.text_frame, text="Bytes: 0 | Words: 0", font=self.bf)
        self.text_bytes_label.pack(anchor="e", padx=6, pady=(0, 6))
        self.secret_textbox.bind("<<Modified>>", self._on_text_change)

        # right: logs & animation
        right = ctk.CTkFrame(frame, width=360)
        right.pack(side="right", fill="y", padx=(8, 12), pady=8)
        right.pack_propagate(False)
        ctk.CTkLabel(right, text="Log", font=self.h2).pack(anchor="w", padx=8, pady=(8, 6))
        self.log_box = ctk.CTkTextbox(right, height=420, font=self.mono)
        self.log_box.pack(fill="both", expand=True, padx=8, pady=(0, 8))
        self.log_box.configure(state="normal")
        self.log("Ready to hide.")
        self.log_box.configure(state="disabled")
        self.anim_label = ctk.CTkLabel(right, text="", font=self.h2)
        self.anim_label.pack(padx=8, pady=(4, 8))

    # -- Extract Tab --
    def _build_extract_tab(self):
        frame = self.tab_extract
        top = ctk.CTkFrame(frame)
        top.pack(fill="both", expand=True, padx=12, pady=8)
        ctk.CTkLabel(top, text="Recover Hidden Data", font=self.h2).pack(anchor="w", pady=(8, 6))
        row = ctk.CTkFrame(top)
        row.pack(fill="x", pady=(0, 8))
        ctk.CTkButton(row, text="Choose Single Stego File", command=self.choose_stego, width=260, font=self.bf).pack(side="left", padx=6)
        ctk.CTkButton(row, text="Choose Multiple Stego Files", command=self.choose_stego_multi, width=260, font=self.bf).pack(side="left", padx=6)

        self.stego_info = ctk.CTkLabel(top, text="No file(s) selected", font=self.bf, wraplength=980, justify="left")
        self.stego_info.pack(pady=(6, 10))

        ctk.CTkLabel(top, text="Password", font=self.h2).pack(pady=(8, 6))
        self.pass_extract = ctk.CTkEntry(top, placeholder_text="Enter password", show="*", width=560, font=self.bf)
        self.pass_extract.pack(pady=8)

        ctk.CTkLabel(top, text="LSB per channel / bits (same as embed)", font=self.h2).pack(pady=(6, 4))
        self.bits_extract_opt = ctk.CTkOptionMenu(top, values=["1", "2", "3", "4"])
        self.bits_extract_opt.set("1")
        self.bits_extract_opt.pack(pady=(0, 8))

        btnrow = ctk.CTkFrame(top)
        btnrow.pack()
        ctk.CTkButton(btnrow, text="Extract → Recover (Single)", command=self.extract_action, width=280, font=self.h2).pack(side="left", padx=6, pady=12)
        ctk.CTkButton(btnrow, text="Extract → Recover (Multiple)", command=self._do_extract_multi_thread, width=280, font=self.h2).pack(side="left", padx=6, pady=12)

        self.extract_log = ctk.CTkTextbox(top, height=340, font=self.mono)
        self.extract_log.pack(fill="both", expand=True, padx=8, pady=8)
        self.extract_log.configure(state="normal")
        self.extract_log.insert("1.0", "Ready to extract...\n")
        self.extract_log.configure(state="disabled")

    # ---- Utility UI functions ----
    def log(self, s: str):
        try:
            self.log_box.configure(state="normal")
            self.log_box.insert("end", time.strftime("[%H:%M:%S] ") + s + "\n")
            self.log_box.see("end")
            self.log_box.configure(state="disabled")
        except Exception:
            pass

    def log_extract(self, s: str):
        try:
            self.extract_log.configure(state="normal")
            self.extract_log.insert("end", time.strftime("[%H:%M:%S] ") + s + "\n")
            self.extract_log.see("end")
            self.extract_log.configure(state="disabled")
        except Exception:
            pass

    def set_status(self, s: str):
        try:
            self.status_label.configure(text=f"Status: {s}")
        except Exception:
            pass

    def _on_pass_change(self, *_):
        pwd = self.pass_entry.get()
        score = 0
        if len(pwd) >= 8: score += 1
        if any(c.isupper() for c in pwd): score += 1
        if any(c.islower() for c in pwd): score += 1
        if any(c.isdigit() for c in pwd): score += 1
        special_chars = set("!@#$%^&*()-_=+[]{}\\|;:'\",<.>/?")
        if any(c in special_chars for c in pwd): score += 1
        labels = {0: "Very weak", 1: "Weak", 2: "Fair", 3: "Good", 4: "Strong", 5: "Very strong"}
        self.pass_strength.configure(text=f"Strength: {labels.get(score, '—')}")

    def _on_text_change(self, *_):
        try:
            self.secret_textbox.edit_modified(False)
        except Exception:
            pass
        txt = self.secret_textbox.get("1.0", "end-1c")
        self.secret_text = txt if txt.strip() else None
        nbytes = len(txt.encode('utf-8'))
        nwords = len(txt.split())
        self.text_bytes_label.configure(text=f"Bytes: {nbytes} | Words: {nwords}")
        if nwords > self.WORD_LIMIT:
            self.text_bytes_label.configure(text=f"Bytes: {nbytes} | Words: {nwords} (OVER LIMIT!)")
        self.update_capacity_info()

    def open_text_editor(self):
        if self.text_frame.winfo_ismapped():
            self.text_frame.pack_forget()
            self.log("Text editor hidden")
        else:
            self.text_frame.pack(fill="both", expand=True, padx=8, pady=(0, 8))
            self.log("Text editor shown")

    def _clear_preview(self):
        self.preview_label.configure(image=None, text="No preview")
        self.preview_label.image = None
        self.cover_path = None
        self.log("Preview cleared")

    # --- File pickers ---
    def choose_cover(self):
        path = filedialog.askopenfilename(title="Select Cover File",
                                          filetypes=[("All files", "*.*"),
                                                     ("Images", "*.png;*.bmp;*.jpg;*.jpeg"),
                                                     ("WAV audio", "*.wav"),
                                                     ("Video", "*.mp4;*.mov;*.avi")])
        if not path:
            return
        self.cover_path = path
        self._show_cover_preview(path)
        self.cover_info.configure(text=f"Cover: {os.path.basename(path)}")
        self.log(f"Selected cover: {path}")
        self.update_capacity_info()

    def choose_secret_file(self):
        path = filedialog.askopenfilename(title="Select Secret File", filetypes=[("All files", "*.*")])
        if not path:
            return
        self.secret_path = path
        self.secret_text = None
        self.secret_textbox.delete("1.0", "end")
        size = os.path.getsize(path)
        self.secret_info.configure(text=f"Secret file: {os.path.basename(path)} ({size} bytes)")
        self.log(f"Selected secret file: {path}")
        self.update_capacity_info()

    def choose_stego(self):
        path = filedialog.askopenfilename(title="Select Stego File", filetypes=[("All files", "*.*")])
        if not path:
            return
        self.stego_path = path
        self.stego_paths_multi = None
        self.stego_info.configure(text=f"Stego (single): {os.path.basename(path)}")
        self.log_extract(f"Selected: {path}")

    def choose_stego_multi(self):
        paths = filedialog.askopenfilenames(title="Select Multiple Stego Files", filetypes=[("Images", "*.png;*.bmp;*.jpg;*.jpeg"), ("All files", "*.*")])
        if not paths:
            return
        self.stego_paths_multi = list(paths)
        self.stego_path = None
        self.stego_info.configure(text=f"Stego (multi): {len(paths)} files selected")
        self.log_extract(f"Selected multiple: {len(paths)} files")

    def _show_cover_preview(self, path: str):
        ext = os.path.splitext(path)[1].lower()
        try:
            if ext in (".png", ".bmp", ".jpg", ".jpeg"):
                img = Image.open(path)
                thumb = img.copy()
                thumb.thumbnail((760, 320))
                img_tk = ImageTk.PhotoImage(thumb)
                self.preview_label.configure(image=img_tk, text="")
                self.preview_label.image = img_tk
            elif ext == ".wav":
                try:
                    data, sr = sf.read(path, dtype='int16')
                    duration = len(data) / sr
                    self.preview_label.configure(text=f"WAV: {os.path.basename(path)}\nDuration: {duration:.2f}s | SR: {sr}", image=None)
                except Exception:
                    self.preview_label.configure(text=f"WAV: {os.path.basename(path)}", image=None)
            elif ext in (".mp4", ".mov", ".avi"):
                if HAS_CV2:
                    cap = cv2.VideoCapture(path)
                    ret, frame = cap.read()
                    fps = cap.get(cv2.CAP_PROP_FPS) or 0
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                    duration = frame_count / fps if fps else 0
                    cap.release()
                    try:
                        if ret:
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            img = Image.fromarray(frame_rgb)
                            img.thumbnail((760, 320))
                            img_tk = ImageTk.PhotoImage(img)
                            self.preview_label.configure(image=img_tk, text=f"Video: {os.path.basename(path)}\nDuration: {duration:.2f}s | FPS: {fps:.1f}")
                            self.preview_label.image = img_tk
                        else:
                            self.preview_label.configure(text=f"Video: {os.path.basename(path)}\nDuration: {duration:.2f}s | FPS: {fps:.1f}", image=None)
                    except Exception:
                        self.preview_label.configure(text=f"Video: {os.path.basename(path)}\nDuration: {duration:.2f}s | FPS: {fps:.1f}", image=None)
                else:
                    self.preview_label.configure(text=f"Video: {os.path.basename(path)}\n(OpenCV not installed)", image=None)
            else:
                self.preview_label.configure(text=f"Cover: {os.path.basename(path)} (append-mode)", image=None)
        except Exception as e:
            self.preview_label.configure(text=f"Preview error: {e}", image=None)

    # ---- Capacity estimation ----
    def update_capacity_info(self, *_):
        if not getattr(self, 'cover_path', None):
            return
        ext = os.path.splitext(self.cover_path)[1].lower()
        payload_size = 0
        name_len = 0
        if self.secret_text:
            payload_size = len(self.secret_text.encode('utf-8'))
        elif self.secret_path:
            payload_size = os.path.getsize(self.secret_path)
            name_len = len(os.path.basename(self.secret_path).encode('utf-8'))
        est = payload_size + 64 + name_len  # rough crypto overhead
        bits = int(self.bits_option.get())
        try:
            if ext in (".png", ".bmp", ".jpg", ".jpeg"):
                img = Image.open(self.cover_path)
                cap = img_capacity_bytes(img, bits_per_channel=bits)
                pct = int(est / cap * 100) if cap > 0 else 0
                self.capacity_label.configure(text=f"Capacity: {cap} bytes | Est: {est} B ({pct}%) | LSBs: {bits}")
            elif ext == ".wav":
                cap = wav_capacity_bytes(self.cover_path, bits_per_sample=bits)
                pct = int(est / cap * 100) if cap > 0 else 0
                self.capacity_label.configure(text=f"Capacity: {cap} bytes | Est: {est} B ({pct}%) | bits/sample: {bits}")
            elif ext in (".mp4", ".mov", ".avi") and HAS_CV2:
                cap = video_capacity_bytes(self.cover_path, bits_per_channel=bits)
                pct = int(est / cap * 100) if cap > 0 else 0
                self.capacity_label.configure(text=f"Capacity: {cap} bytes | Est: {est} B ({pct}%) | bits/ch: {bits}")
            else:
                self.capacity_label.configure(text=f"Append-mode. Est payload: {est} B")
        except Exception:
            pass

    # ---- Embed flow ----
    def embed_action(self):
        t = threading.Thread(target=self._do_embed, daemon=True)
        t.start()

    def _do_embed(self):
        try:
            self.progress.set(0.0)
            if not getattr(self, 'cover_path', None):
                messagebox.showerror("Error", "Select a cover file.")
                return
            if not (getattr(self, 'secret_path', None) or getattr(self, 'secret_text', None)):
                messagebox.showerror("Error", "Select a secret file or enter secret text.")
                return
            pwd = self.pass_entry.get().strip()
            if not pwd:
                messagebox.showerror("Error", "Enter a strong password.")
                return
            # prepare payload
            if getattr(self, 'secret_text', None):
                words = self.secret_text.strip().split()
                if len(words) > self.WORD_LIMIT:
                    messagebox.showerror("Error", f"Secret text exceeds {self.WORD_LIMIT} word limit.")
                    return
                ptype = 0
                name = ""
                plain = self.secret_text.encode('utf-8')
            else:
                ptype = 1
                name = os.path.basename(self.secret_path)
                with open(self.secret_path, 'rb') as f:
                    plain = f.read()

            self.log("Encrypting payload with AES-256-GCM...")
            enc_blob = encrypt_bytes(pwd, plain)
            packed = pack_payload(ptype, name, enc_blob)  # MAGIC + header + enc_blob
            wrapped = pack_single_part(packed)  # transport wrapper

            ext = os.path.splitext(self.cover_path)[1].lower()
            bits = int(self.bits_option.get())

            # multi-image split (image-only)
            if ext in (".png", ".bmp", ".jpg", ".jpeg") and self.multi_split_var.get():
                base_img = Image.open(self.cover_path)
                if base_img.mode not in ("RGB", "RGBA"):
                    base_img = base_img.convert("RGBA")
                first_cap = img_capacity_bytes(base_img, bits_per_channel=bits)
                out_base = filedialog.asksaveasfilename(defaultextension=".png", title="Save first stego image as...")
                if not out_base:
                    return
                capacities = [first_cap]
                extra_imgs = []
                remaining = len(wrapped) - first_cap
                out_paths = [out_base]
                if remaining > 0:
                    messagebox.showinfo("Multi-image split", "Payload larger than first image capacity. Select additional images.")
                    extra_paths = filedialog.askopenfilenames(title="Select additional images", filetypes=[("Images", "*.png;*.bmp;*.jpg;*.jpeg")])
                    if not extra_paths:
                        messagebox.showerror("Insufficient capacity", "No extra images selected.")
                        return
                    for idx, p in enumerate(extra_paths, start=2):
                        im = Image.open(p)
                        if im.mode not in ("RGB", "RGBA"):
                            im = im.convert("RGBA")
                        cap = img_capacity_bytes(im, bits_per_channel=bits)
                        capacities.append(cap)
                        base, ext_out = os.path.splitext(out_base)
                        out_paths.append(f"{base}_part{idx:02d}{ext_out}")
                        extra_imgs.append(p)
                chunks = split_payload_into_chunks(wrapped, capacities)
                for i, chunk in enumerate(chunks):
                    wrapped_chunk = wrap_part(i + 1, len(chunks), chunk)
                    def cb_pct(pct, idx=i):
                        self.progress.set(pct / 100)
                    if i == 0:
                        embed_in_image(self.cover_path, wrapped_chunk, out_paths[0], bits_per_channel=bits, progress_callback=lambda p: cb_pct(p))
                    else:
                        embed_in_image(extra_imgs[i - 1], wrapped_chunk, out_paths[i], bits_per_channel=bits, progress_callback=lambda p: cb_pct(p))
                    self.log(f"Embedded part {i + 1}/{len(chunks)} -> {out_paths[i]}")
                self.set_status("Done — embedded across multiple images")
                self._anim_access(True)
                messagebox.showinfo("Done", f"Created {len(out_paths)} stego images.\nFirst: {out_paths[0]}")
                return

            out = filedialog.asksaveasfilename(defaultextension=os.path.splitext(self.cover_path)[1], title="Save stego as...", filetypes=[("All files", "*.*")])
            if not out:
                return

            def progress_cb(pct):
                try:
                    self.progress.set(pct / 100)
                except Exception:
                    pass

            if ext in (".png", ".bmp", ".jpg", ".jpeg"):
                self.log("Embedding into image (LSB)...")
                embed_in_image(self.cover_path, wrapped, out, bits_per_channel=bits, progress_callback=progress_cb)
            elif ext == ".wav":
                self.log("Embedding into WAV (LSB)...")
                embed_in_wav(self.cover_path, wrapped, out, bits_per_sample=bits, progress_callback=progress_cb)
            elif ext in (".mp4", ".mov", ".avi") and HAS_CV2:
                self.log("Embedding into video (LSB)...")
                embed_in_video(self.cover_path, wrapped, out, bits_per_channel=bits, progress_callback=progress_cb)
            else:
                self.log("Embedding by append (fallback)...")
                embed_by_append(self.cover_path, wrapped, out, progress_callback=progress_cb)

            self.log(f"Embedded successfully -> {out}")
            self.set_status("Access Granted — Embedded")
            self._anim_access(True)
            messagebox.showinfo("Success", f"Hidden successfully: {out}")

        except Exception as e:
            self.log(f"Embed error: {e}")
            self.set_status("Access Denied — Error")
            self._anim_access(False)
            messagebox.showerror("Embed failed", str(e))

    # ---- Extract flow ----
    def extract_action(self):
        t = threading.Thread(target=self._do_extract_single, daemon=True)
        t.start()

    def _do_extract_single(self):
        try:
            self.progress.set(0.0)
            path = filedialog.askopenfilename(title="Select Stego File", filetypes=[("All files", "*.*")])
            if not path:
                return
            pwd = self.pass_entry.get().strip()
            if not pwd:
                messagebox.showerror("Error", "Enter password.")
                return
            ext = os.path.splitext(path)[1].lower()
            bits = int(self.bits_extract_opt.get())
            raw = None
            try:
                if ext in (".png", ".bmp", ".jpg", ".jpeg"):
                    self.log("Reading image LSB payload...")
                    raw = extract_from_image(path, bits_per_channel=bits, progress_callback=lambda p: self.progress.set(p / 100))
                elif ext == ".wav":
                    self.log("Reading WAV LSB payload...")
                    raw = extract_from_wav(path, bits_per_sample=bits, progress_callback=lambda p: self.progress.set(p / 100))
                elif ext in (".mp4", ".mov", ".avi") and HAS_CV2:
                    self.log("Reading video LSB payload...")
                    raw = extract_from_video(path, bits_per_channel=bits, progress_callback=lambda p: self.progress.set(p / 100))
                else:
                    self.log("Trying append-mode extraction...")
                    raw = extract_by_append(path)
                    if raw is None:
                        try:
                            raw = extract_from_image(path, bits_per_channel=bits, progress_callback=lambda p: self.progress.set(p / 100))
                        except Exception:
                            raw = None
            except Exception as e:
                raw = None
                self.log(f"Extraction read error: {e}")

            if raw is None:
                self.set_status("Access Denied — No stego found")
                self._anim_access(False)
                messagebox.showerror("Error", "No stego payload found.")
                return

            # check if single-part or split-part
            if raw.startswith(SPLIT_TAG):
                part = unwrap_part(raw)
                if not part:
                    self._err_header()
                    return
                _, _, chunk = part
                buffer = chunk
            else:
                try:
                    buffer = unwrap_single_part(raw)
                except Exception:
                    self._err_header()
                    return

            # now buffer should be the packed payload (MAGIC + header + enc_blob)
            meta = unpack_payload(buffer)
            if not meta:
                self._err_header("Invalid payload header")
                return
            ptype, name, enc_blob = meta
            dec = decrypt_bytes(pwd, enc_blob)
            if dec is None:
                self.set_status("Access Denied — Wrong password or corrupted")
                self._anim_access(False)
                messagebox.showerror("Error", "Wrong password or data corrupted.")
                return
            # success
            self.set_status("Access Granted — Decrypted")
            self._anim_access(True)
            if ptype == 0:  # text
                try:
                    text = dec.decode('utf-8', errors='replace')
                except Exception:
                    text = dec.decode('latin-1', errors='replace')
                self.log_extract("Extracted TEXT (displaying)...")
                win = ctk.CTkToplevel(self.root)
                win.title("Extracted Secret Text")
                win.geometry("900x640")
                txt = ctk.CTkTextbox(win, font=self.bf)
                txt.pack(fill="both", expand=True, padx=8, pady=8)
                txt.insert("1.0", text)
                txt.configure(state="disabled")
                # also save to file
                out_path = os.path.join(OUTPUT_DIR, f"extracted_text_{int(time.time())}.txt")
                with open(out_path, 'wb') as f:
                    f.write(dec)
                self.log_extract(f"Text saved to: {out_path}")
                messagebox.showinfo("Success", f"Text extracted and saved: {out_path}")
            else:  # file
                suggested = name if name else f"extracted_file_{int(time.time())}"
                out_path = os.path.join(OUTPUT_DIR, suggested)
                # if name existed, do not overwrite silently — add timestamp
                if os.path.exists(out_path):
                    base, extn = os.path.splitext(out_path)
                    out_path = f"{base}_{int(time.time())}{extn}"
                with open(out_path, 'wb') as f:
                    f.write(dec)
                self.log_extract(f"Extracted file saved: {out_path}")
                messagebox.showinfo("Success", f"File extracted and saved: {out_path}")

        except Exception as e:
            self.log_extract(f"Unexpected extract error: {e}")
            messagebox.showerror("Error", str(e))

    def _do_extract_multi_thread(self):
        t = threading.Thread(target=self._do_extract_multi, daemon=True)
        t.start()

    def _do_extract_multi(self):
        try:
            self.progress.set(0.0)
            if not self.stego_paths_multi:
                messagebox.showerror("Error", "Select multiple stego files (use Extract tab).")
                return
            pwd = self.pass_entry.get().strip()
            if not pwd:
                messagebox.showerror("Error", "Enter password.")
                return
            bits = int(self.bits_extract_opt.get())
            parts = []
            total_files = len(self.stego_paths_multi)
            for idx, p in enumerate(self.stego_paths_multi, start=1):
                ext = os.path.splitext(p)[1].lower()
                raw = None
                try:
                    if ext in (".png", ".bmp", ".jpg", ".jpeg"):
                        raw = extract_from_image(p, bits_per_channel=bits, progress_callback=lambda p_, i=idx: self.progress.set(((i - 1) / total_files) + p_ / 100 / total_files))
                    else:
                        raw = extract_by_append(p)
                except Exception as e:
                    self.log_extract(f"Read error {p}: {e}")
                if not raw:
                    continue
                if not raw.startswith(SPLIT_TAG):
                    try:
                        buffer = unwrap_single_part(raw)
                        # single full payload in one file -> decrypt & output
                        meta = unpack_payload(buffer)
                        if meta:
                            ptype, name, enc_blob = meta
                            dec = decrypt_bytes(pwd, enc_blob)
                            if dec is not None:
                                # write & done
                                if ptype == 0:
                                    out_path = os.path.join(OUTPUT_DIR, f"extracted_text_{int(time.time())}.txt")
                                    with open(out_path, 'wb') as f:
                                        f.write(dec)
                                    self.log_extract(f"Text saved to: {out_path}")
                                    messagebox.showinfo("Success", f"Text extracted and saved: {out_path}")
                                    return
                                else:
                                    suggested = name if name else f"extracted_file_{int(time.time())}"
                                    out_path = os.path.join(OUTPUT_DIR, suggested)
                                    with open(out_path, 'wb') as f:
                                        f.write(dec)
                                    self.log_extract(f"Extracted file saved: {out_path}")
                                    messagebox.showinfo("Success", f"File extracted and saved: {out_path}")
                                    return
                    except Exception:
                        continue
                part = unwrap_part(raw)
                if part:
                    parts.append(part)
            if not parts:
                self._err_header("No valid split parts found.")
                return
            parts.sort(key=lambda x: x[0])
            total = parts[0][1]
            if len(parts) != total:
                self._err_header(f"Expected {total} parts but got {len(parts)}.")
                return
            merged = b"".join(ch for (_, _, ch) in parts)
            # merged should be wrapped single-part or direct packed
            try:
                buffer = unwrap_single_part(merged)
            except Exception:
                buffer = merged
            meta = unpack_payload(buffer)
            if not meta:
                self._err_header("Merged payload invalid.")
                return
            ptype, name, enc_blob = meta
            dec = decrypt_bytes(pwd, enc_blob)
            if dec is None:
                self._err_header("Wrong password or corrupted merged payload.")
                return
            if ptype == 0:
                out_path = os.path.join(OUTPUT_DIR, f"extracted_text_{int(time.time())}.txt")
                with open(out_path, 'wb') as f:
                    f.write(dec)
                self.log_extract(f"Text saved to: {out_path}")
                messagebox.showinfo("Success", f"Text extracted and saved: {out_path}")
            else:
                suggested = name if name else f"extracted_file_{int(time.time())}"
                out_path = os.path.join(OUTPUT_DIR, suggested)
                with open(out_path, 'wb') as f:
                    f.write(dec)
                self.log_extract(f"Extracted file saved: {out_path}")
                messagebox.showinfo("Success", f"File extracted and saved: {out_path}")

        except Exception as e:
            self.log_extract(f"Multi-extract error: {e}")
            messagebox.showerror("Error", str(e))

    def _err_header(self, msg="Invalid header"):
        self.set_status(f"Access Denied — {msg}")
        self.log_extract(f"Header error: {msg}")
        self._anim_access(False)
        messagebox.showerror("Error", msg)

    # ---- visual access animation ----
    def _anim_access(self, granted: bool):
        color = "#22c55e" if granted else "#ef4444"
        text = "ACCESS GRANTED" if granted else "ACCESS DENIED"
        self.anim_label.configure(text=text, text_color=color)
        def blink(i=0):
            if i > 6:
                self.anim_label.configure(text="")
                return
            if i % 2 == 0:
                self.anim_label.configure(text=text)
            else:
                self.anim_label.configure(text="")
            self.root.after(220, blink, i + 1)
        blink()


# Entry point
def main():
    root = ctk.CTk()
    app = StegoUniversalApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
