# ===== 1단계 — 라벨용 컨택트시트 만들기 =====
# Colab 새 노트북에 이 셀 하나만 붙여넣고 실행. GPU 불필요. 1~2분.
# 업로드: 프레임 zip 파일들 (영상마다 하나씩, 여러 개 한꺼번에 가능)
#
# zip 안의 폴더 이름이 영상 이름이 됨. Windows 압축의 역슬래시 경로도 처리함.
# 출력: 영상별 컨택트시트 PNG. 내려받아 보면서 사람이 보이는 프레임 번호를 적으면 됨.
#
# **YOLO 를 돌리기 전에 라벨을 만드는 것이 중요함.**
# 모델이 그린 박스를 먼저 보면 "모델이 놓친 사람"을 사람 눈도 같이 놓치게 됨.

import os, glob, zipfile, shutil
from PIL import Image, ImageDraw, ImageFont
from google.colab import files

COLS, ROWS = 6, 5          # 시트 한 장에 30칸
CELL_W = 320               # 칸 가로 픽셀. 사람이 작게 나오면 480으로 올릴 것
ROOT = '/content/frames'

# --- 자료를 어디서 읽을지 -------------------------------------------------
# 30MB 업로드 한도를 넘는 zip 이 있으므로 기본은 구글 드라이브 연결임.
# 드라이브 '내 드라이브' 안에 person_frames 폴더를 만들고 zip 을 넣어 둘 것.
USE_DRIVE = True
DRIVE_DIR = '/content/drive/MyDrive/person_frames'
ONLY = []                  # 특정 zip 만 쓰려면 이름을 적음. 비우면 전부
# ---------------------------------------------------------------------------

shutil.rmtree(ROOT, ignore_errors=True)
os.makedirs(ROOT, exist_ok=True)

if USE_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
    zips = sorted(glob.glob(f'{DRIVE_DIR}/*.zip'))
    assert zips, f'{DRIVE_DIR} 에서 zip 을 찾지 못했습니다'
    if ONLY:
        zips = [z for z in zips if os.path.basename(z) in ONLY]
    print('읽을 zip:', [os.path.basename(z) for z in zips])
else:
    zips = [k for k in files.upload() if k.lower().endswith('.zip')]

for n in zips:
    with zipfile.ZipFile(n) as z:
        for info in z.infolist():
            if info.is_dir():
                continue
            # Windows Compress-Archive 가 역슬래시로 쓰는 경우를 정리
            rel = info.filename.replace('\\', '/')
            if not rel.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            parts = [p for p in rel.split('/') if p not in ('', '.', '..')]
            vid = parts[-2] if len(parts) >= 2 else os.path.splitext(n)[0]
            dst = os.path.join(ROOT, vid)
            os.makedirs(dst, exist_ok=True)
            with z.open(info) as src, open(os.path.join(dst, parts[-1]), 'wb') as out:
                shutil.copyfileobj(src, out)

vids = sorted(d for d in os.listdir(ROOT) if os.path.isdir(os.path.join(ROOT, d)))
assert vids, 'zip 안에서 이미지를 찾지 못했습니다'

try:
    FONT = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 34)
except Exception:
    FONT = ImageFont.load_default()

os.makedirs('/content/sheets', exist_ok=True)
manifest = []

for vid in vids:
    paths = sorted(glob.glob(f'{ROOT}/{vid}/*'))
    # 프레임 번호는 이 정렬 순서 기준의 1-based. 라벨을 적을 때도 이 번호를 씀.
    with open(f'/content/sheets/{vid}_order.txt', 'w') as fp:
        for i, p in enumerate(paths, 1):
            fp.write(f'{i}\t{os.path.basename(p)}\n')

    w0, h0 = Image.open(paths[0]).size
    cell_h = round(CELL_W * h0 / w0)
    per = COLS * ROWS
    n_sheet = (len(paths) + per - 1) // per

    for s in range(n_sheet):
        chunk = paths[s * per:(s + 1) * per]
        sheet = Image.new('RGB', (COLS * CELL_W, ROWS * cell_h), (24, 24, 24))
        d = ImageDraw.Draw(sheet)
        for j, p in enumerate(chunk):
            im = Image.open(p).convert('RGB').resize((CELL_W, cell_h))
            x, y = (j % COLS) * CELL_W, (j // COLS) * cell_h
            sheet.paste(im, (x, y))
            idx = s * per + j + 1
            d.rectangle([x, y, x + 78, y + 44], fill=(0, 0, 0))
            d.text((x + 8, y + 4), str(idx), fill=(255, 220, 0), font=FONT)
            d.rectangle([x, y, x + CELL_W - 1, y + cell_h - 1], outline=(90, 90, 90))
        out = f'/content/sheets/{vid}__{s + 1:02d}.png'
        sheet.save(out, quality=88)
    manifest.append((vid, len(paths), n_sheet))
    print(f'{vid:<34} 프레임 {len(paths):>4}장 · 시트 {n_sheet}장')

print('\n총', sum(m[1] for m in manifest), '프레임')
print('시트를 내려받아 확인한 뒤, 사람이 보이는 프레임 번호를 아래 형식으로 적으면 됩니다.')
print("  예)  '논현중_튀김': '1-14,22,30-45,101'")

shutil.make_archive('/content/sheets_pack', 'zip', '/content/sheets')
files.download('/content/sheets_pack.zip')
