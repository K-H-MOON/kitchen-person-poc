# ===== 사각지대 확인용 컨택트시트 =====
# "사람 없음으로 라벨했고 모델도 탐지하지 않은" 377프레임만 골라 시트로 묶음.
#
# 왜 필요한가 — 라벨 오류를 오탐 이미지에서 찾았는데, 오탐은 정의상 모델이 탐지한
# 프레임임. 따라서 그 방식으로는 "모델도 놓치고 라벨도 놓친" 프레임을 발견할 수 없음.
# 이 구간을 확인해야 인식률이 낙관적으로 치우쳐 있는지 알 수 있음.
#
# 준비 — 드라이브 person_frames 폴더의 프레임 zip (이미 올려 둔 것 그대로)
# Colab 새 노트북에 이 셀 하나. GPU 불필요. 3~5분.

import os, glob, zipfile, shutil
from PIL import Image, ImageDraw, ImageFont
from google.colab import drive, files

DRIVE_DIR = '/content/drive/MyDrive/person_frames'
COLS, ROWS, CW = 5, 6, 460        # 시트 한 장에 30칸. 구석의 작은 것을 봐야 하므로 크게

# yolov8x · conf 0.25 기준으로 계산한 사각지대 프레임 번호
BLIND = {
    'gaewon_cctv_in': '18-79,83-85,87,90,93-97,102-110,112,114-116,118-120,122-126,'
                      '129-133,135,139,141-144,146,148-155,157-159,162-168,170,'
                      '172-173,204-205,219-240',
    'wonchon_fry_full': '20,24,38-42,50-51,53-54,58-63,65-70,72-74,76,79,81-82,84-85,'
                        '88-90,92,94-95,97-98,100-101,103-108,111,115-117,121,126-152,'
                        '154,157-161,163-164,183,185-215,218-219,224-240',
    'np_inhwa_stir': '7-10,18-22,25,27,32-40,53-55,59-66,69-70,72-81',
    'np_sunggok_stir': '3,9-11,14,19,31-32,34,44,52-53,57-61,63-64',
    'yeongdong_soup': '1,21,29,67-70,73,135,137-138,146-147,152,171',
    'sunggok_soup_in1': '7,10-11,36-38,45',
    'ulsan_stir_in': '47-49',
    'robot_fry_in': '33',
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

for v in BLIND:
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

os.makedirs('/content/blind_sheets', exist_ok=True)
total = 0
for v in sorted(BLIND):
    paths = sorted(glob.glob(f'{ROOT}/{v}/*'))
    idx = sorted(parse(BLIND[v]))
    pick = [(i, paths[i - 1]) for i in idx if i <= len(paths)]
    assert len(pick) == len(idx), f'{v}: 프레임 수 불일치 {len(paths)} vs 최대 {max(idx)}'
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
        sheet.save(f'/content/blind_sheets/BLIND_{v}__{s // per + 1:02d}.jpg', quality=86)
    total += len(pick)
    print(f'{v:<22} {len(pick):>4}장 → 시트 {(len(pick) + per - 1) // per}장')

print(f'{"합계":<22} {total:>4}장')
shutil.make_archive('/content/blind_sheets', 'zip', '/content/blind_sheets')
print(f'-> blind_sheets.zip  {os.path.getsize("/content/blind_sheets.zip") / 1e6:.1f}MB')
files.download('/content/blind_sheets.zip')
