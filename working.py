#!/usr/bin/env python3
"""
game_to_colors_and_audio.py

Converte arquivo ou pasta (pasta -> zip) em:
 - PNG (cada pixel = 3 bytes do arquivo)
 - colors.txt (uma linha #RRGGBB por pixel)
 - WAV (8-bit unsigned PCM) contendo bytes do arquivo
Também reconstrói o arquivo original a partir do PNG ou do WAV.
Salva metadados em <basename>.meta.json.

Recomendações: para pastas grandes, atente-se ao tamanho da imagem/WAV gerados.
"""
import io
import math
import json
import zlib
import zipfile
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
from PIL import Image
import wave
import os
import shutil
import tempfile

# -----------------------
# UTILS
# -----------------------
def get_safe_output_folder():
    base = Path.home() / "Pictures" / "AudioGameConverter"
    base.mkdir(parents=True, exist_ok=True)
    return base

def zip_folder(folder_path: Path) -> Path:
    """Zip the folder into a temporary file and return path to zip."""
    tmp = Path(tempfile.mkdtemp())
    zip_path = tmp / (folder_path.name + ".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in folder_path.rglob("*"):
            zf.write(p, p.relative_to(folder_path))
    return zip_path

# -----------------------
# CORE: bytes <-> pixels
# -----------------------
def bytes_to_pixels(data: bytes):
    pad = (-len(data)) % 3
    if pad:
        data = data + b'\x00' * pad
    arr = np.frombuffer(data, dtype=np.uint8)
    pixels = arr.reshape((-1, 3))
    return pixels, pad

def pixels_to_bytes(pixels: np.ndarray, pad: int):
    b = pixels.astype(np.uint8).tobytes()
    if pad:
        return b[:-pad]
    return b

def pixels_to_image_file(pixels: np.ndarray, out_path: Path, width: int = None):
    n = pixels.shape[0]
    if width is None:
        width = int(math.ceil(math.sqrt(n)))
    height = int(math.ceil(n / width))
    canvas = np.zeros((height * width, 3), dtype=np.uint8)
    canvas[:n, :] = pixels
    canvas = canvas.reshape((height, width, 3))
    img = Image.fromarray(canvas, mode="RGB")
    img.save(out_path, format="PNG")
    return width, height

def image_file_to_pixels(img_path: Path):
    img = Image.open(img_path).convert("RGB")
    arr = np.array(img, dtype=np.uint8)
    h, w, _ = arr.shape
    flat = arr.reshape((-1, 3))
    return flat, w, h

def pixels_to_hex_lines(pixels: np.ndarray):
    return ['#{:02x}{:02x}{:02x}'.format(r,g,b) for (r,g,b) in pixels.tolist()]

# -----------------------
# ENCODE / DECODE
# -----------------------
def prepare_input(path: Path):
    """If path is folder, zip it; return (data_bytes, is_zip, temp_zip_path_or_None)."""
    if path.is_dir():
        zip_path = zip_folder(path)
        data = zip_path.read_bytes()
        return data, True, zip_path
    else:
        return path.read_bytes(), False, None

def encode_file(path: Path, out_folder: Path, use_zlib: bool = False):
    orig_name = path.name
    data, was_zipped, zip_temp = prepare_input(path)
    used_zlib = False
    if use_zlib:
        comp = zlib.compress(data)
        if len(comp) < len(data):
            data = comp
            used_zlib = True
    pixels, pad = bytes_to_pixels(data)

    base = out_folder / (path.stem)
    base.mkdir(parents=True, exist_ok=True)

    img_path = base / (path.stem + "_colors.png")
    hex_path = base / (path.stem + "_colors.txt")
    wav_path = base / (path.stem + "_audio.wav")
    meta_path = base / (path.stem + ".meta.json")

    w,h = pixels_to_image_file(pixels, img_path)
    hex_lines = pixels_to_hex_lines(pixels)
    hex_path.write_text("\n".join(hex_lines), encoding='utf-8')

    # write WAV: 8-bit unsigned PCM (simple mapping of bytes -> samples)
    # WAV sampwidth 1 expects unsigned bytes (0..255). Good for reversible mapping.
    with wave.open(str(wav_path), 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(1)   # 8-bit
        wf.setframerate(44100)  # sample rate arbitrary; user can change later
        wf.writeframes(pixels.flatten().tobytes())

    # save meta
    meta = {
        "orig_name": orig_name,
        "orig_size_bytes": (zip_temp.stat().st_size if zip_temp else path.stat().st_size),
        "was_zipped": bool(was_zipped),
        "zlib_used": used_zlib,
        "pad_bytes": pad,
        "image_width": w,
        "image_height": h,
        "files": {
            "png": str(img_path),
            "hex": str(hex_path),
            "wav": str(wav_path),
            "meta": str(meta_path)
        }
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding='utf-8')

    # cleanup temp zip folder if any
    if zip_temp:
        try:
            shutil.rmtree(zip_temp.parent)
        except Exception:
            pass

    return img_path, hex_path, wav_path, meta_path

def reconstruct_from_image(img_path: Path, meta_path: Path, out_folder: Path):
    pixels, w, h = image_file_to_pixels(img_path)
    # load meta
    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    pad = int(meta.get("pad_bytes", 0))
    zlib_used = bool(meta.get("zlib_used", False))
    was_zipped = bool(meta.get("was_zipped", False))
    orig_name = meta.get("orig_name", "reconstructed.bin")

    raw = pixels_to_bytes(pixels, pad)
    if zlib_used:
        raw = zlib.decompress(raw)
    out_name = out_folder / orig_name
    out_name.write_bytes(raw)

    # if was_zipped and original was a folder, the reconstructed file is a zip:
    # user can unzip it manually
    return out_name

def reconstruct_from_wav(wav_path: Path, meta_path: Path, out_folder: Path):
    # read wav raw frames
    with wave.open(str(wav_path), 'rb') as r:
        frames = r.readframes(r.getnframes())
    meta = json.loads(meta_path.read_text(encoding='utf-8'))
    pad = int(meta.get("pad_bytes", 0))
    zlib_used = bool(meta.get("zlib_used", False))
    was_zipped = bool(meta.get("was_zipped", False))
    orig_name = meta.get("orig_name", "reconstructed.bin")

    # frames are bytes already; note that in encode we wrote pixels.flatten() which equals frames
    raw = frames
    # remove padding if any (we don't have pad info maybe; meta stored pad)
    if pad:
        raw = raw[:-pad]
    if zlib_used:
        raw = zlib.decompress(raw)
    out_name = out_folder / orig_name
    out_name.write_bytes(raw)
    return out_name

# -----------------------
# GUI
# -----------------------
def gui_encode():
    path = filedialog.askopenfilename(title="Selecione arquivo (ou selecione uma pasta)", initialdir=str(Path.home()))
    # allow folder selection via askdirectory if user wants; check if empty string then ask directory
    if not path:
        # try directory selection
        path_dir = filedialog.askdirectory(title="Ou selecione uma pasta")
        if not path_dir:
            return
        path = path_dir
    p = Path(path)
    use_z = var_zlib.get()
    out_folder = get_safe_output_folder()
    try:
        img, hexf, wavf, meta = encode_file(p, out_folder, use_z)
        messagebox.showinfo("Encode OK", f"Salvo em:\n{img}\n{hexf}\n{wavf}\n{meta}")
    except Exception as e:
        messagebox.showerror("Erro", str(e))

def gui_reconstruct_from_image():
    img_path = filedialog.askopenfilename(title="Selecione PNG gerado pelo encoder", filetypes=[("PNG","*.png")])
    if not img_path:
        return
    meta_path = Path(img_path).with_suffix(".meta.json")
    if not meta_path.exists():
        # try to prompt user for meta
        mp = filedialog.askopenfilename(title="Selecione o arquivo .meta.json correspondente", filetypes=[("JSON","*.json")])
        if not mp:
            messagebox.showerror("Erro", "Não encontrou meta. Preciso do .meta.json para reconstruir.")
            return
        meta_path = Path(mp)
    out_folder = get_safe_output_folder()
    try:
        out_file = reconstruct_from_image(Path(img_path), meta_path, out_folder)
        messagebox.showinfo("Reconstruído", f"Arquivo restaurado em:\n{out_file}")
    except Exception as e:
        messagebox.showerror("Erro", str(e))

def gui_reconstruct_from_wav():
    wav_path = filedialog.askopenfilename(title="Selecione WAV gerado pelo encoder", filetypes=[("WAV","*.wav")])
    if not wav_path:
        return
    meta_path = Path(wav_path).with_suffix(".meta.json")
    if not meta_path.exists():
        mp = filedialog.askopenfilename(title="Selecione o arquivo .meta.json correspondente", filetypes=[("JSON","*.json")])
        if not mp:
            messagebox.showerror("Erro", "Não encontrou meta. Preciso do .meta.json para reconstruir.")
            return
        meta_path = Path(mp)
    out_folder = get_safe_output_folder()
    try:
        out_file = reconstruct_from_wav(Path(wav_path), meta_path, out_folder)
        messagebox.showinfo("Reconstruído", f"Arquivo restaurado em:\n{out_file}")
    except Exception as e:
        messagebox.showerror("Erro", str(e))

# -----------------------
# MAIN GUI SETUPa
# -----------------------
root = tk.Tk()
root.title("Game ↔ Colors & Audio Converter")
root.geometry("420x260")

tk.Label(root, text="Game ↔ Colors & Audio", font=("Segoe UI", 16)).pack(pady=10)

frame = tk.Frame(root)
frame.pack(pady=8)

btn_encode = tk.Button(frame, text="Encode file/folder → PNG + colors + WAV", width=40, command=gui_encode)
btn_encode.grid(row=0, column=0, padx=8, pady=6, columnspan=2)

var_zlib = tk.BooleanVar(value=False)
chk_zlib = tk.Checkbutton(frame, text="Apply zlib compression before encoding (try if file is large)", variable=var_zlib)
chk_zlib.grid(row=1, column=0, columnspan=2, pady=4)

btn_recon_img = tk.Button(frame, text="Reconstruct original from PNG", width=40, command=gui_reconstruct_from_image)
btn_recon_img.grid(row=2, column=0, padx=8, pady=6, columnspan=2)

btn_recon_wav = tk.Button(frame, text="Reconstruct original from WAV", width=40, command=gui_reconstruct_from_wav)
btn_recon_wav.grid(row=3, column=0, padx=8, pady=6, columnspan=2)

tk.Label(root, text="Output folder: Pictures/AudioGameConverter", font=("Segoe UI", 9)).pack(pady=8)

root.mainloop()
