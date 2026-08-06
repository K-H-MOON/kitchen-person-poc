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
    # '논현중_튀김': '1-14,22,30-45,101',
    # '개원중cctv_튀김투입': '',
}
# 시점 메모. 결과를 시점별로 묶어 보기 위한 것이며 비워 둬도 동작함.
VIEWPOINT = {
    # '논현중_튀김': '핸드헬드·눈높이',
    # '개원중cctv_튀김투입': '고정 CCTV·천장',
}
MODELS = ['yolov8n.pt', 'yolov8s.pt', 'yolov8x.pt']
CONFS = [0.10, 0.25, 0.50]
MAIN_CONF = 0.25          # 표에 쓸 기준 임계값
FPS = 1.0                 # 프레임을 뽑은 간격. 공백을 초로 환산할 때 씀
# ---------------------------------------------------------------------------

!pip -q install ultralytics==8.3.*

import os, glob, zipfile, shutil, json
from collections import defaultdict
from ultralytics import YOLO

ROOT = '/content/frames'
shutil.rmtree(ROOT, ignore_errors=True); os.makedirs(ROOT, exist_ok=True)
from google.colab import files
up = files.upload()

for n in [k for k in up if k.lower().endswith('.zip')]:
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
def score(mp, v, th):
    pos = gt[v]
    b = raw[mp][v]
    P = [i for i in range(1, len(b) + 1) if i in pos]
    N = [i for i in range(1, len(b) + 1) if i not in pos]
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

    ok1 = (sum(recs) / len(recs) if recs else 0) >= 0.80
    ok2 = worst <= 1
    if mp == MODELS[-1]:
        print(f'\n  사전 등록 판정 ({mp} · conf {MAIN_CONF})')
        print(f'    조건 1  인식률 80% 이상        : {"통과" if ok1 else "실패"}')
        print(f'    조건 2  연속 미탐 1프레임 이하 : {"통과" if ok2 else "실패"}'
              f'  (최악 {worst}프레임)')

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
