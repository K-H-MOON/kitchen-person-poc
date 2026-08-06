# ===== 2단계 — 기성 YOLO person 측정 =====
# 컨택트시트로 라벨을 다 적은 뒤에 실행. GPU 켤 것(런타임 > 런타임 유형 변경 > T4).
# 업로드: 1단계와 똑같은 프레임 zip 파일들
#
# 파인튜닝 없이 COCO 사전학습 가중치를 그대로 씀. 회의 액션 아이템 3의 1차 답임.
# 모델 크기 세 종류를 같이 재는 이유 — 안 잡히는 것이 "모델이 작아서"인지
# "우리 시점이 특이해서"인지 갈라야 하기 때문임.

# ---------------------------------------------------------------------------
# 여기에 라벨을 붙여넣을 것. 키는 zip 안의 폴더 이름과 정확히 같아야 함.
# 값은 사람이 보이는 프레임 번호. 범위는 '30-45', 낱개는 '22', 없으면 '' 로 둠.
LABELS = {
    # --- 검수 완료 ---
    'gaewon_cctv_in': '1-17,80-82,91-92,98-101,113,174-200,207-217',   # 240장 · 사람 65
    'yeongdong_soup': '15-20,22-26,28,31-66,74-130,133-134,136,'
                      '149-150,153-170,176-185,198,209,233-239,272-286',  # 286장 · 사람 162
    'sunggok_soup_in1': '47-50,60-64,69-73,75-77,81-89',                # 91장 · 사람 26
    'ulsan_stir_in': '1-46,50-61',                                      # 61장 · 사람 58
    'robot_fry_in': '1,3,4',                                            # 33장 · 사람 3 (전부 미세)
    'wonchon_fry_full': '1-15,21-23,46-47,56-57,77,153,155-156,216',    # 240장 · 사람 27
    'np_sunggok_stir': '',                                              # 71장 · 사람 0
    'np_inhwa_stir': '',                                                # 81장 · 사람 0
}

# 사람이 '일부만 겨우 보이는' 프레임. LABELS 의 부분집합이어야 함.
# 앞뒤 프레임과 비교해야 판별되는 것들. 인식률을 두 무리로 나눠 냄.
PERSON_PARTIAL = {
    'gaewon_cctv_in': '91-92,98-101,113,174,200',
    'yeongdong_soup': '15,20,74,130,136,153,198,209',
    'sunggok_soup_in1': '60,69,75',
    'robot_fry_in': '1,3,4',
    'wonchon_fry_full': '10-15,46-47,56-57,77,153,155-156,216',
}
# 촬영자 본인의 몸이 찍힌 프레임 (1인칭 시점). LABELS 의 부분집합이어야 함.
# 사람인 것은 맞으나 로봇에 붙은 고정 카메라에서는 생기지 않는 상황이므로,
# 포함한 수치와 제외한 수치를 나란히 냄. 라벨 자체는 바꾸지 않음.
FIRSTPERSON = {
    'wonchon_fry_full': '21-23',
}

# 시점 메모. 결과를 시점별로 묶어 보기 위한 것이며 비워 둬도 동작함.
VIEWPOINT = {
    # 'np_sunggok_stir': '핸드헬드·로봇전용',
    # 'gaewon_cctv_in': '고정 CCTV·천장',
}

# 로봇 팔이 '일부만 보이거나 근접·사각으로 잡힌' 프레임 번호.
# 팔 전체가 보이는 프레임과 나눠서 오탐이 어디에 몰리는지 봄.
# 비워 두면 이 분석을 건너뜀.
ARM_PARTIAL = {
    # 'np_sunggok_stir': '35,41-42,65-71',
    # 'np_inhwa_stir': '',
}
MODELS = ['yolov8n.pt', 'yolov8s.pt', 'yolov8x.pt']
CONFS = [0.10, 0.25, 0.50]
MAIN_CONF = 0.25          # 표에 쓸 기준 임계값
FPS = 1.0                 # 프레임을 뽑은 간격. 공백을 초로 환산할 때 씀

# 자료 위치 — 1단계와 같게 맞출 것
USE_DRIVE = True
DRIVE_DIR = '/content/drive/MyDrive/person_frames'
ONLY = []                 # 1차로 6개만 돌리려면 여기에 zip 이름을 적음
# ---------------------------------------------------------------------------

!pip -q install ultralytics==8.3.*

import os, glob, zipfile, shutil, json
from collections import defaultdict
from ultralytics import YOLO

ROOT = '/content/frames'
shutil.rmtree(ROOT, ignore_errors=True); os.makedirs(ROOT, exist_ok=True)
from google.colab import files

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
            rel = info.filename.replace('\\', '/')
            if not rel.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            parts = [p for p in rel.split('/') if p not in ('', '.', '..')]
            vid = parts[-2] if len(parts) >= 2 else os.path.splitext(n)[0]
            os.makedirs(f'{ROOT}/{vid}', exist_ok=True)
            with z.open(info) as src, open(f'{ROOT}/{vid}/{parts[-1]}', 'wb') as out:
                shutil.copyfileobj(src, out)


def parse(spec):
    """'1-14,22,30-45' -> {1..14, 22, 30..45}"""
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


vids = sorted(d for d in os.listdir(ROOT) if os.path.isdir(f'{ROOT}/{d}'))
missing = [v for v in vids if v not in LABELS]
if missing:
    print('!! 라벨이 없는 폴더:', missing)
    print('   LABELS 의 키를 폴더 이름과 똑같이 맞출 것. 이 폴더들은 건너뜁니다.\n')
vids = [v for v in vids if v in LABELS]
assert vids, '라벨과 이름이 맞는 폴더가 없습니다'

gt, order = {}, {}
for v in vids:
    order[v] = sorted(glob.glob(f'{ROOT}/{v}/*'))
    pos = parse(LABELS[v])
    bad = [i for i in pos if i < 1 or i > len(order[v])]
    assert not bad, f'{v}: 프레임 번호 범위를 벗어남 {bad[:5]} (총 {len(order[v])}장)'
    stray = parse(PERSON_PARTIAL.get(v, '')) - pos
    assert not stray, f'{v}: PERSON_PARTIAL 이 LABELS 밖의 번호를 가리킴 {sorted(stray)[:5]}'
    stray = parse(FIRSTPERSON.get(v, '')) - pos
    assert not stray, f'{v}: FIRSTPERSON 이 LABELS 밖의 번호를 가리킴 {sorted(stray)[:5]}'
    gt[v] = pos
    print(f'{v:<34} {len(order[v]):>4}장 · 사람있음 {len(pos):>4} · 사람없음 {len(order[v]) - len(pos):>4}')

# --- 추론 ---------------------------------------------------------------
raw = defaultdict(dict)      # raw[model][vid] = [프레임별 person conf 최대값]
for mp in MODELS:
    m = YOLO(mp)
    for v in vids:
        best = []
        for i in range(0, len(order[v]), 32):
            for r in m.predict(order[v][i:i + 32], conf=0.05, classes=[0],
                               imgsz=640, verbose=False):
                c = r.boxes.conf.cpu().numpy()
                best.append(float(c.max()) if len(c) else 0.0)
        raw[mp][v] = best
    print(f'{mp} 완료')

# --- 집계 ---------------------------------------------------------------
def score(mp, v, th, drop_fp1=False):
    """drop_fp1=True 면 촬영자 본인 몸 프레임을 계산에서 통째로 뺌"""
    pos = gt[v]
    b = raw[mp][v]
    skip = parse(FIRSTPERSON.get(v, '')) if drop_fp1 else set()
    P = [i for i in range(1, len(b) + 1) if i in pos and i not in skip]
    N = [i for i in range(1, len(b) + 1) if i not in pos and i not in skip]
    rec = sum(b[i - 1] >= th for i in P) / len(P) if P else None
    fp = sum(b[i - 1] >= th for i in N) / len(N) if N else None
    return rec, fp


def pct(x):
    return '  -  ' if x is None else f'{x * 100:5.1f}%'


def maxgap(mp, v, th):
    """사람이 보이는 구간 안에서 연속으로 놓친 최대 프레임 수.
    연속 k프레임을 놓치면 실제 탐지 공백은 (k+1)/FPS 초임."""
    b = raw[mp][v]
    pos = gt[v]
    run = best = 0
    for i in range(1, len(b) + 1):
        if i in pos and b[i - 1] < th:
            run += 1
            best = max(best, run)
        else:
            run = 0
    return best


for mp in MODELS:
    print('\n' + '=' * 88)
    print(f'{mp}  ·  conf {MAIN_CONF}')
    print('=' * 88)
    print(f'{"영상":<30}{"시점":<16}{"인식률":>8}{"오탐률":>8}{"최장공백":>10}  (있음/없음)')
    for v in vids:
        rec, fp = score(mp, v, MAIN_CONF)
        g = maxgap(mp, v, MAIN_CONF)
        vp = VIEWPOINT.get(v, '')
        gap_s = f'{(g + 1) / FPS:.0f}초' if g else '-'
        print(f'{v:<30}{vp:<16}{pct(rec):>8}{pct(fp):>8}{gap_s:>10}  '
              f'({len(gt[v])}/{len(order[v]) - len(gt[v])})')
    # 영상별 평균(macro) — 프레임 수가 많은 영상 하나에 끌려가지 않게
    recs = [r for v in vids for r, _ in [score(mp, v, MAIN_CONF)] if r is not None]
    fps = [f for v in vids for _, f in [score(mp, v, MAIN_CONF)] if f is not None]
    worst = max((maxgap(mp, v, MAIN_CONF) for v in vids), default=0)
    print('-' * 88)
    print(f'{"영상별 평균(macro) / 최악 공백":<46}'
          f'{pct(sum(recs) / len(recs) if recs else None):>8}'
          f'{pct(sum(fps) / len(fps) if fps else None):>8}'
          f'{(str(round((worst + 1) / FPS)) + "초") if worst else "-":>10}')

    if any(FIRSTPERSON.get(v) for v in vids):
        r2 = [r for v in vids for r, _ in [score(mp, v, MAIN_CONF, True)] if r is not None]
        f2 = [f for v in vids for _, f in [score(mp, v, MAIN_CONF, True)] if f is not None]
        print(f'{"└ 촬영자 본인 몸 제외":<46}'
              f'{pct(sum(r2) / len(r2) if r2 else None):>8}'
              f'{pct(sum(f2) / len(f2) if f2 else None):>8}')

    ok1 = (sum(recs) / len(recs) if recs else 0) >= 0.80
    ok2 = worst <= 1
    if mp == MODELS[-1]:
        print(f'\n  사전 등록 판정 ({mp} · conf {MAIN_CONF})')
        print(f'    조건 1  인식률 80% 이상        : {"통과" if ok1 else "실패"}')
        print(f'    조건 2  연속 미탐 1프레임 이하 : {"통과" if ok2 else "실패"}'
              f'  (최악 {worst}프레임)')

# --- 사람 노출 정도별 인식률 ---------------------------------------------
# 전체가 보이는 사람과 일부만 겨우 보이는 사람을 나눠 냄.
# 회의에서 연구 포인트로 지목된 "머리만 보이는 시점·조리복에 가린 형체"가 여기 걸림.
if any(PERSON_PARTIAL.get(v) for v in vids):
    print('\n' + '=' * 88)
    print(f'사람 노출 정도별 인식률  ·  conf {MAIN_CONF}')
    print('=' * 88)
    print(f'{"모델":<14}{"영상":<22}{"확실히 보임":>16}{"일부만":>16}')
    for mp in MODELS:
        for v in vids:
            spec = PERSON_PARTIAL.get(v)
            if not spec:
                continue
            part = parse(spec)
            b = raw[mp][v]
            full_i = [i for i in sorted(gt[v]) if i not in part]
            part_i = [i for i in sorted(gt[v]) if i in part]
            def rec(idx):
                if not idx:
                    return '  -  (0)'
                r = sum(b[i - 1] >= MAIN_CONF for i in idx) / len(idx)
                return f'{r * 100:5.1f}% ({len(idx)})'
            print(f'{mp:<14}{v:<22}{rec(full_i):>16}{rec(part_i):>16}')

# --- 로봇 팔 노출 정도별 오탐 -------------------------------------------
# 가설: 팔 전체가 보이면 구조가 드러나 사람과 구분되고,
#       잘리거나 각도가 틀어지면 형체 정보가 사라져 사람으로 읽힘.
if any(ARM_PARTIAL.get(v) for v in vids):
    print('\n' + '=' * 88)
    print(f'로봇 팔 노출 정도별 오탐  ·  conf {MAIN_CONF}')
    print('=' * 88)
    print(f'{"모델":<14}{"영상":<22}{"전체 보임":>16}{"일부만/사각":>16}')
    for mp in MODELS:
        for v in vids:
            spec = ARM_PARTIAL.get(v)
            if spec is None:
                continue
            part = parse(spec)
            b = raw[mp][v]
            # 사람 없음 프레임만 대상 (오탐 계산이므로)
            neg = [i for i in range(1, len(b) + 1) if i not in gt[v]]
            full_i = [i for i in neg if i not in part]
            part_i = [i for i in neg if i in part]
            def rate(idx):
                if not idx:
                    return '  -  (0)'
                r = sum(b[i - 1] >= MAIN_CONF for i in idx) / len(idx)
                return f'{r * 100:5.1f}% ({len(idx)})'
            print(f'{mp:<14}{v:<22}{rate(full_i):>16}{rate(part_i):>16}')

print('\n' + '=' * 88)
print('임계값별 macro 인식률 / 오탐률')
print('=' * 88)
print(f'{"모델":<14}' + ''.join(f'{"conf " + str(c):>20}' for c in CONFS))
for mp in MODELS:
    row = f'{mp:<14}'
    for c in CONFS:
        recs = [r for v in vids for r, _ in [score(mp, v, c)] if r is not None]
        fps = [f for v in vids for _, f in [score(mp, v, c)] if f is not None]
        r = sum(recs) / len(recs) if recs else None
        f = sum(fps) / len(fps) if fps else None
        row += f'{pct(r) + " / " + pct(f):>20}'
    print(row)

# --- 눈으로 볼 것: 미탐과 오탐 이미지를 뽑아 둠 ----------------------------
BIG = MODELS[-1]
os.makedirs('/content/look/miss', exist_ok=True)
os.makedirs('/content/look/fp', exist_ok=True)
m = YOLO(BIG)
n_miss = n_fp = 0
for v in vids:
    b = raw[BIG][v]
    for i, p in enumerate(order[v], 1):
        hit = b[i - 1] >= MAIN_CONF
        want = i in gt[v]
        if hit == want:
            continue
        kind = 'fp' if hit else 'miss'
        r = m.predict(p, conf=MAIN_CONF, classes=[0], imgsz=640, verbose=False)[0]
        r.save(filename=f'/content/look/{kind}/{v}__{i:04d}.jpg')
        n_miss += kind == 'miss'; n_fp += kind == 'fp'
print(f'\n미탐 {n_miss}장 · 오탐 {n_fp}장을 /content/look 에 저장했습니다.')

json.dump({'labels': LABELS, 'viewpoint': VIEWPOINT, 'conf_main': MAIN_CONF,
           'raw': {k: v for k, v in raw.items()},
           'counts': {v: len(order[v]) for v in vids}},
          open('/content/person_baseline.json', 'w'), ensure_ascii=False)
shutil.make_archive('/content/look_pack', 'zip', '/content/look')
files.download('/content/person_baseline.json')
files.download('/content/look_pack.zip')
