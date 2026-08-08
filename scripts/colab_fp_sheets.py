# ===== 오탐 전수 확인용 컨택트시트 =====
# 1차 측정에서 나온 오탐 385장·미탐 28장을 영상별로 전부 시트로 묶음.
# 앞 30장만 확인했던 것을 전수로 바꾸기 위한 것임.
#
# 준비 — look_pack.zip 을 구글 드라이브 person_frames 폴더에 올려 둘 것 (131MB)
# Colab 새 노트북에 이 셀 하나. GPU 불필요. 2~3분.

import os, glob, zipfile, shutil
from PIL import Image, ImageDraw, ImageFont
from google.colab import drive, files

DRIVE_DIR = '/content/drive/MyDrive/person_frames'
COLS, ROWS, CW = 5, 8, 400        # 시트 한 장에 40칸

drive.mount('/content/drive')
src = f'{DRIVE_DIR}/look_pack.zip'
assert os.path.exists(src), f'{src} 가 없습니다. look_pack.zip 을 드라이브에 올릴 것'

shutil.rmtree('/content/look', ignore_errors=True)
zipfile.ZipFile(src).extractall('/content/look')

try:
    F = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 24)
except Exception:
    F = ImageFont.load_default()


def sheets(paths, tag, outdir):
    """40장씩 묶어 시트 여러 장으로 저장"""
    if not paths:
        return 0
    w0, h0 = Image.open(paths[0]).size
    ch = round(CW * h0 / w0)
    per = COLS * ROWS
    n = 0
    for s in range(0, len(paths), per):
        chunk = paths[s:s + per]
        rows = (len(chunk) + COLS - 1) // COLS
        sheet = Image.new('RGB', (COLS * CW, rows * ch), (20, 20, 20))
        d = ImageDraw.Draw(sheet)
        for j, p in enumerate(chunk):
            im = Image.open(p).convert('RGB').resize((CW, ch))
            x, y = (j % COLS) * CW, (j // COLS) * ch
            sheet.paste(im, (x, y))
            name = os.path.basename(p)[:-4]
            label = name.split('__')[-1]          # 프레임 번호만
            d.rectangle([x, y, x + 96, y + 32], fill=(0, 0, 0))
            d.text((x + 5, y + 3), label, fill=(255, 220, 0), font=F)
            d.rectangle([x, y, x + CW - 1, y + ch - 1], outline=(80, 80, 80))
        n += 1
        sheet.save(f'{outdir}/{tag}__{n:02d}.jpg', quality=84)
    return n


os.makedirs('/content/fp_sheets', exist_ok=True)

print('=' * 60)
print('오탐 (사람이 없는데 person 박스가 나온 프레임)')
print('=' * 60)
vids = sorted({os.path.basename(p).split('__')[0] for p in glob.glob('/content/look/fp/*.jpg')})
total = 0
for v in vids:
    ps = sorted(glob.glob(f'/content/look/fp/{v}__*.jpg'))
    k = sheets(ps, f'FP_{v}', '/content/fp_sheets')
    total += len(ps)
    print(f'{v:<22} {len(ps):>4}장 → 시트 {k}장')
print(f'{"합계":<22} {total:>4}장')

miss = sorted(glob.glob('/content/look/miss/*.jpg'))
k = sheets(miss, 'MISS', '/content/fp_sheets')
print(f'\n미탐 {len(miss)}장 → 시트 {k}장')

shutil.make_archive('/content/fp_sheets', 'zip', '/content/fp_sheets')
sz = os.path.getsize('/content/fp_sheets.zip') / 1e6
print(f'\n-> fp_sheets.zip  {sz:.1f}MB')
files.download('/content/fp_sheets.zip')
