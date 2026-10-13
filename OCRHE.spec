# -*- mode: python ; coding: utf-8 -*-
import os
from PyInstaller.utils.hooks import collect_all

datas = [
    ("data\\corpus_lite.txt", "data"),
    ("data\\config.json", "data"),
    ("data\\fixes_general.tsv", "data"),
    ("data\\.lm_counts.pkl", "data"),
    ("data\\.lm_bigrams.pkl", "data"),
    ("data\\.lm_trigrams.pkl", "data"),
    ("models\\base_synth.pt", "models"),
    ("models\\general.pt", "models"),
    ("models\\typed.pt", "models"),
    ("icon.ico", "."),
]
binaries = []
hiddenimports = [
    "paths", "charset", "model", "synth", "dataset", "train", "segment",
    "infer", "make_template", "postfix", "gates", "fixes", "records",
    "ai_correction", "app", "pypdfium2.raw",
]
for package in ("torch", "pypdfium2", "certifi", "truststore", "keyring"):
    package_datas, package_binaries, package_hiddenimports = collect_all(package)
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hiddenimports

a = Analysis(
    ["cli.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=["pyi_rth_torch_dll.py"],
    excludes=[],
    noarchive=False,
    optimize=0,
)

# Keep the same current MSVC runtime used by the GUI bundle. PyInstaller can
# otherwise pick up an incompatible copy from PATH, preventing torch from
# loading c10.dll in the one-file executable.
system_dir = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32")
required_crt = ("msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll")
for name in required_crt:
    source = os.path.join(system_dir, name)
    if not os.path.isfile(source):
        raise FileNotFoundError(f"Required Windows runtime DLL is missing: {source}")

crt_names = (
    "msvcp140.dll",
    "msvcp140_1.dll",
    "msvcp140_2.dll",
    "msvcp140_atomic_wait.dll",
    "msvcp140_codecvt_ids.dll",
    "vcruntime140.dll",
    "vcruntime140_1.dll",
    "vcruntime140_threads.dll",
)
for name in crt_names:
    source = os.path.join(system_dir, name)
    if os.path.isfile(source):
        a.binaries[:] = [
            entry for entry in a.binaries
            if os.path.basename(entry[0]).lower() != name.lower()
        ]
        a.binaries.append((name, source, "BINARY"))

exe = EXE(
    PYZ(a.pure),
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="OCRHE",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
