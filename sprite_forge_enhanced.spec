# -*- mode: python ; coding: utf-8 -*-

block_cipher = None


a = Analysis(['sprite_forge_enhanced.py'],
             pathex=[],
             binaries=[],
             datas=[],
             hiddenimports=['PyQt6.sip'],
             hookspath=[],
             hooksconfig={},
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)

exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='Sprite Forge Enhanced',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=False,
          disable_windowed_traceback=False,
          target_arch=None,
          codesign_identity=None,
          entitlements_file=None , icon='icons/app_icon.ico')
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='Sprite Forge Enhanced')
app = BUNDLE(coll,
             name='Sprite Forge Enhanced.app',
             icon='icons/app_icon.icns',
             bundle_identifier=None)
