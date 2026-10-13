# -*- mode: python ; coding: utf-8 -*-
import os
from PyInstaller.utils.hooks import collect_all, collect_data_files

embedded_browser = os.environ.get('KTAVYADAI_EMBEDDED_BROWSER') == '1'
datas = [
    ('data\\corpus_lite.txt', 'data'),
    ('data\\config.json', 'data'),
    ('data\\fixes_general.tsv', 'data'),
    ('data\\.lm_counts.pkl', 'data'),
    ('data\\.lm_bigrams.pkl', 'data'),
    ('data\\.lm_trigrams.pkl', 'data'),
    ('models\\base_synth.pt', 'models'),
    ('models\\general.pt', 'models'),
    ('models\\typed.pt', 'models'),
    ('icon.ico', '.'),
]
binaries = []
hiddenimports = [
    'paths', 'charset', 'model', 'synth', 'dataset', 'train', 'segment',
    'infer', 'make_template', 'postfix', 'gates', 'fixes', 'records',
    'ai_correction', 'app',
]
packages = ['torch', 'pypdfium2', 'certifi', 'truststore', 'keyring']
if embedded_browser:
    packages.append('webview')
for package in packages:
    package_datas, package_binaries, package_hiddenimports = collect_all(package)
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hiddenimports
if embedded_browser:
    datas += collect_data_files('webview')
    hiddenimports += [
        'webview.platforms.qt',
        'qtpy',
        'qtpy.QtCore',
        'qtpy.QtGui',
        'qtpy.QtWidgets',
        'qtpy.QtWebChannel',
        'qtpy.QtWebEngineCore',
        'qtpy.QtWebEngineWidgets',
        'PyQt6.QtCore',
        'PyQt6.QtGui',
        'PyQt6.QtWidgets',
        'PyQt6.QtWebChannel',
        'PyQt6.QtWebEngineCore',
        'PyQt6.QtWebEngineWidgets',
        'PyQt6.QtWebEngineQuick',
    ]


a = Analysis(
    ['main.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=['pyi_rth_torch_dll.py'],
    excludes=[],
    noarchive=False,
    optimize=0,
)

# PyInstaller can discover an outdated CRT through PATH (for example, from a
# JDK installation). Bundle the current Windows redistributable instead.
system_dir = os.path.join(os.environ.get('SystemRoot', r'C:\Windows'), 'System32')
required_crt = ('msvcp140.dll', 'vcruntime140.dll', 'vcruntime140_1.dll')
for name in required_crt:
    source = os.path.join(system_dir, name)
    if not os.path.isfile(source):
        raise FileNotFoundError(f'Required Windows runtime DLL is missing: {source}')

crt_names = (
    'msvcp140.dll',
    'msvcp140_1.dll',
    'msvcp140_2.dll',
    'msvcp140_atomic_wait.dll',
    'msvcp140_codecvt_ids.dll',
    'vcruntime140.dll',
    'vcruntime140_1.dll',
    'vcruntime140_threads.dll',
)
for name in crt_names:
    source = os.path.join(system_dir, name)
    if os.path.isfile(source):
        a.binaries[:] = [
            entry for entry in a.binaries
            if os.path.basename(entry[0]).lower() != name.lower()
        ]
        a.binaries.append((name, source, 'BINARY'))

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='KtavYadAI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    icon='icon.ico',
    version='version_info.txt',
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
