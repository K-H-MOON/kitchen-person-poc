# ===== 노출 정도 재분류용 컨택트시트 =====
# "사람 있음"으로 확정된 350프레임만 모아 시트로 묶음.
#
# 왜 필요한가 — 지금 확실/일부만 분류가 '1-17번은 확실'처럼 구간 단위로 붙어 있음.
# 사람이 화면에 들어오고 나가는 구간에서는 같은 범위 안에서도 노출 정도가 달라지므로,
# 구간 라벨은 확실 쪽에 일부만을 섞어 넣음. 미탐 31장 중 18장이 확실로 분류돼 있었고
# 그 설명이 대부분 "구석에 일부만 보임"이었음.
#
# 현재 분류는 시트에 표시하지 않음. 보고 판정하면 기존 판정을 따라가게 됨.
#
# 준비 — 드라이브 person_frames 폴더의 프레임 zip (이미 올려 둔 것 그대로)
# Colab 새 노트북에 이 셀 하나. GPU 불필요. 3~5분.

import os, glob, zipfile, shutil
from PIL import Image, ImageDraw, ImageFont
from google.colab import drive, files

DRIVE_DIR = '/content/drive/MyDrive/person_frames'
COLS, ROWS, CW = 5, 6, 460

# 사람 있음으로 확정된 프레임 (2026-08-07 전수 확인 반영)
PERSON = {
    'gaewon_cctv_in': '1-17,80-82,91-92,98-101,113,174-200,207-217',
    'yeongdong_soup': '11-13,15-20,22-26,28,30,31-66,74-130,133-134,136,144-147,152,'
                      '149-150,153-170,176-185,198,209,233-239,272-286',
    'sunggok_soup_in1': '47-50,60-64,69-73,75-77,81-89',
    'ulsan_stir_in': '1-46,50-61',
    'wonchon_fry_full': '1-15,21-23,46-47,56-57,77,153,155-156,216',
    'robot_fry_in': '1,3,4',
}


def parse(spec):
    s = set()
    for tok in str(spec).replace(' ', '').split(','):
        if not tok:
            continue
        if '-' in tok:
            a, b = tok.split('-')
            s.update(range(int(a), int(b) + 1))
        else:
            s.add(int(tok))
    return s


drive.mount('/content/drive')
ROOT = '/content/frames'
shutil.rmtree(ROOT, ignore_errors=True); os.makedirs(ROOT, exist_ok=True)

for v in PERSON:
    z = f'{DRIVE_DIR}/{v}.zip'
    assert os.path.exists(z), f'{z} 없음'
    with zipfile.ZipFile(z) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            rel = info.filename.replace('\\', '/')
            if not rel.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            name = [p for p in rel.split('/') if p][-1]
            os.makedirs(f'{ROOT}/{v}', exist_ok=True)
            with zf.open(info) as src, open(f'{ROOT}/{v}/{name}', 'wb') as out:
                shutil.copyfileobj(src, out)

try:
    F = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 30)
except Exception:
    F = ImageFont.load_default()

os.makedirs('/content/person_sheets', exist_ok=True)
total = 0
for v in sorted(PERSON):
    paths = sorted(glob.glob(f'{ROOT}/{v}/*'))
    idx = sorted(parse(PERSON[v]))
    assert max(idx) <= len(paths), f'{v}: 프레임 부족 {len(paths)} < {max(idx)}'
    pick = [(i, paths[i - 1]) for i in idx]
    w0, h0 = Image.open(pick[0][1]).size
    ch = round(CW * h0 / w0)
    per = COLS * ROWS
    for s in range(0, len(pick), per):
        chunk = pick[s:s + per]
        rows = (len(chunk) + COLS - 1) // COLS
        sheet = Image.new('RGB', (COLS * CW, rows * ch), (20, 20, 20))
        d = ImageDraw.Draw(sheet)
        for j, (num, p) in enumerate(chunk):
            im = Image.open(p).convert('RGB').resize((CW, ch))
            x, y = (j % COLS) * CW, (j // COLS) * ch
            sheet.paste(im, (x, y))
            d.rectangle([x, y, x + 92, y + 40], fill=(0, 0, 0))
            d.text((x + 6, y + 4), str(num), fill=(255, 220, 0), font=F)
            d.rectangle([x, y, x + CW - 1, y + ch - 1], outline=(80, 80, 80))
        sheet.save(f'/content/person_sheets/PERSON_{v}__{s // per + 1:02d}.jpg', quality=86)
    total += len(pick)
    print(f'{v:<22} {len(pick):>4}장 → 시트 {(len(pick) + per - 1) // per}장')

print(f'{"합계":<22} {total:>4}장')
shutil.make_archive('/content/person_sheets', 'zip', '/content/person_sheets')
print(f'-> person_sheets.zip  {os.path.getsize("/content/person_sheets.zip") / 1e6:.1f}MB')
files.download('/content/person_sheets.zip')
