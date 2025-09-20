# soccer.app.py — v0.5
# (API-FOOTBALL + form(H/A) + auto-refresh + odds + lineup-event weighting + calibration
#  + xG/Poisson score probs + formation matchup weights)

import math
import os
from datetime import date
from dateutil.parser import parse as parse_dt

import requests
import pandas as pd
import altair as alt
import streamlit as st

# ----------------------------
# 기본 설정 / 버전
# ----------------------------
st.set_page_config(page_title="축구 승률 예측 – API-FOOTBALL", layout="centered")
APP_VERSION = "v0.5 (events+calibration+xG+poisson+formation)"
st.title("⚽ 승률 예측 – API-FOOTBALL 연동")
st.caption("경기 조회 → 라인업/이벤트(시간가중/교체) → 폼(H/A) → 배당(암시확률/캘리브) → 승률 + 스코어 분포")
st.caption(f"현재 파일: {__file__} • 버전: {APP_VERSION}")

# ----------------------------
# 키/상수
# ----------------------------
API_FOOTBALL_KEY = st.secrets.get("APIFOOTBALL_KEY") or os.getenv("APIFOOTBALL_KEY")
if not API_FOOTBALL_KEY:
    st.warning("`.streamlit/secrets.toml`에 APIFOOTBALL_KEY 를 넣어주세요.")

BASE = "https://v3.football.api-sports.io"
LEAGUES = {"Premier League": 39, "La Liga": 140, "Serie A": 135, "Bundesliga": 78, "Ligue 1": 61}
AUTO_REFRESH_SEC = 20  # 실시간 토글 ON일 때 자동 새로고침 간격(초)
AVG_GOALS = 2.60       # 리그 평균 득점(간이 상수)

# 임시 가중치(튜닝 대상)
BETA = {
    "bias": 0.0,
    "home_adv": 0.35,
    "form_diff": 0.80,
    "injured_starters": -0.40,
    "attack_rating_diff": 1.10,
    "defense_rating_diff": 0.70,
    "odds_signal": 1.20,
}

# 포지션별 라인업 가중치 (간이)
POS_WEIGHTS = {
    "GK": {"atk": 0.1, "dfn": 1.6},
    "DF": {"atk": 0.2, "dfn": 1.0},   # CB/LB/RB
    "MF": {"atk": 0.6, "dfn": 0.6},   # CM/DM/WM
    "FW": {"atk": 1.2, "dfn": 0.2},   # ST/W/AM
    "UNK": {"atk": 0.3, "dfn": 0.3},
}

# 간단한 포메이션 매칭 가중치(실전용은 데이터 기반으로 확장 권장)
# 값은 홈 기준 (atk, dfn) 델타; 원정은 반대 부호 적용
FORMATION_MATCHUPS = {
    ("4-3-3", "3-5-2"): (+0.08, +0.03),
    ("4-3-3", "4-4-2"): (+0.04,  0.00),
    ("3-5-2", "4-3-3"): (-0.05, -0.02),
    ("4-2-3-1", "3-5-2"): (+0.05, +0.02),
    ("4-2-3-1", "5-4-1"): (-0.03, +0.02),
    ("5-3-2", "4-3-3"): (-0.04, +0.05),
    # 기본값은 (0,0)
}

# ----------------------------
# 유틸 함수
# ----------------------------
def sigmoid(z: float) -> float:
    return 1 / (1 + math.exp(-z))

def logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p/(1-p))

def win_prob_with_beta(feats_dict, beta_odds):
    z = (
        BETA["bias"]
        + BETA["home_adv"] * feats_dict.get("home_adv", 0)
        + BETA["form_diff"] * feats_dict.get("form_diff", 0)
        + BETA["injured_starters"] * feats_dict.get("injured_starters", 0)
        + BETA["attack_rating_diff"] * feats_dict.get("attack_rating_diff", 0)
        + BETA["defense_rating_diff"] * feats_dict.get("defense_rating_diff", 0)
        + beta_odds * feats_dict.get("odds_signal", 0)
    )
    return 1 / (1 + math.exp(-z))

# xG 추정(간이): 피처를 로짓으로 묶어 λ로 변환
def estimate_expected_goals(home_feats, away_feats, base_goals=AVG_GOALS):
    # 점수 분배 비율 r ∈ (0,1): 홈이 더 강할수록 r↑
    # 공격/수비 차 + 홈이점 + 폼을 압축한 스코어
    score_home = (1.0*home_feats.get("attack_rating_diff",0)
                  + 0.6*home_feats.get("defense_rating_diff",0)
                  + 0.5*home_feats.get("form_diff",0)
                  + 0.4*home_feats.get("home_adv",0))
    score_away = (1.0*(-home_feats.get("attack_rating_diff",0))
                  + 0.6*(-home_feats.get("defense_rating_diff",0))
                  + 0.5*(-home_feats.get("form_diff",0)))
    r = sigmoid(0.8*(score_home - score_away))  # 0~1
    lam_home = max(0.05, base_goals * r)
    lam_away = max(0.05, base_goals * (1 - r))
    return lam_home, lam_away

def poisson_pmf(lmbd: float, k: int) -> float:
    return math.exp(-lmbd) * (lmbd ** k) / math.factorial(k)

def score_matrix(lh: float, la: float, gmax: int = 6):
    rows = []
    for h in range(gmax+1):
        for a in range(gmax+1):
            p = poisson_pmf(lh, h) * poisson_pmf(la, a)
            rows.append({"home": h, "away": a, "prob": p})
    df = pd.DataFrame(rows)
    return df

def derive_markets_from_score(df_scores: pd.DataFrame):
    p_home = df_scores.query("home>away")["prob"].sum()
    p_draw = df_scores.query("home==away")["prob"].sum()
    p_away = df_scores.query("home<away")["prob"].sum()
    btts = df_scores.query("home>0 and away>0")["prob"].sum()
    over25 = df_scores.query("home+away>=3")["prob"].sum()
    under25 = 1 - over25
    top3 = df_scores.sort_values("prob", ascending=False).head(3)
    return p_home, p_draw, p_away, btts, over25, under25, top3

# ----------------------------
# API-FOOTBALL helpers
# ----------------------------
def _headers():
    return {"x-apisports-key": API_FOOTBALL_KEY}

@st.cache_data(ttl=1800)  # 30분 캐시
def af_get(path: str, params=None):
    r = requests.get(BASE + path, headers=_headers(), params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    if data.get("errors"):
        raise RuntimeError(str(data["errors"]))
    return data.get("response", [])

@st.cache_data(ttl=3600)
def af_list_fixtures(league_id: int, season: int, the_date: str):
    return af_get("/fixtures", {"league": league_id, "season": season, "date": the_date})

@st.cache_data(ttl=600)
def af_lineups(fixture_id: int):
    return af_get("/fixtures/lineups", {"fixture": fixture_id})

@st.cache_data(ttl=60)
def af_events(fixture_id: int):
    return af_get("/fixtures/events", {"fixture": fixture_id})

@st.cache_data(ttl=300)
def af_odds_by_fixture(fixture_id: int):
    return af_get("/odds", {"fixture": fixture_id})

# 최근 폼
@st.cache_data(ttl=600)
def af_team_last_fixtures(team_id: int, last: int = 5):
    resp = af_get("/fixtures", {"team": team_id, "last": last + 6})
    fins = [f for f in resp if f["fixture"]["status"]["short"] in ("FT", "AET", "PEN")]
    return fins

def pick_last_completed(fixtures: list, count: int = 5):
    return fixtures[:count] if fixtures else []

def is_home_for_team(fix, team_id: int) -> bool:
    return fix["teams"]["home"]["id"] == team_id

def points_per_game(fixtures: list, team_id: int) -> float:
    if not fixtures: return 0.0
    pts = 0
    for f in fixtures:
        home_id = f["teams"]["home"]["id"]
        hg, ag = f["goals"]["home"], f["goals"]["away"]
        if hg == ag:
            pts += 1
        else:
            winner_home = hg > ag
            if (winner_home and team_id == home_id) or ((not winner_home) and team_id != home_id):
                pts += 3
    return round(pts / len(fixtures), 3)

# 라인업 가중치 계산(포지션별)
def pos_bucket(pos_raw: str) -> str:
    p = (pos_raw or "").upper()
    if p in ("G", "GK"): return "GK"
    if any(k in p for k in ["CB","LB","RB","DF","CBR","CBL"]): return "DF"
    if any(k in p for k in ["CM","DM","MF","WM","RM","LM"]):   return "MF"
    if any(k in p for k in ["FW","ST","AM","LW","RW","CF"]):   return "FW"
    return "UNK"

def calc_team_ratings(lineup_obj):
    if not lineup_obj: return 0.0, 0.0
    start = lineup_obj.get("startXI", [])
    atk = dfn = 0.0
    for p in start:
        pos = p["player"].get("pos") or p["player"].get("position") or ""
        bucket = pos_bucket(pos)
        atk += POS_WEIGHTS[bucket]["atk"]
        dfn += POS_WEIGHTS[bucket]["dfn"]
    return atk/11.0, dfn/11.0

# ==== 이벤트 가중치 유틸 ====
def minute_remaining_weight(ev) -> float:
    """남은 시간 비율(0~1). 간단화(90분 기준)."""
    t = ev.get("time", {}) or {}
    m = t.get("elapsed") or 0
    et = t.get("extra") or 0
    minute = max(0, min(100, (m or 0) + (et or 0)))
    return max(0.0, (90 - minute) / 90.0)

def build_player_bucket_index(lineup_obj):
    idx = {}
    if not lineup_obj:
        return idx
    for sect in ["startXI", "substitutes"]:
        for p in lineup_obj.get(sect, []):
            pl = p.get("player", {}) or {}
            pid = pl.get("id")
            pos = pl.get("pos") or pl.get("position") or ""
            if pid:
                idx[pid] = pos_bucket(pos)
    return idx

def bucket_weights_delta(in_bucket: str, out_bucket: str):
    kin = POS_WEIGHTS.get(in_bucket or "UNK", POS_WEIGHTS["UNK"])
    kout = POS_WEIGHTS.get(out_bucket or "UNK", POS_WEIGHTS["UNK"])
    scale = 0.8 / 11.0
    return (scale * (kin["atk"] - kout["atk"]), scale * (kin["dfn"] - kout["dfn"]))

# ----------------------------
# 사이드바: 리그/시즌/날짜 + 캘리브 α
# ----------------------------
with st.sidebar:
    st.header("⚙️ 데이터 선택")
    season = st.number_input("시즌(연도)", min_value=2015, max_value=2030, value=date.today().year, step=1)
    league_name = st.selectbox("리그", list(LEAGUES.keys()), index=0)
    league_id = LEAGUES[league_name]
    the_date = st.date_input("날짜", value=date.today())
    fetch_btn = st.button("경기 불러오기", use_container_width=True)

    st.divider()
    cal_alpha = st.slider(
        "캘리브레이션 α(배당 혼합)", 0.0, 1.0, 0.30, 0.05,
        help="최종 홈승 확률 = (1-α)*모델 + α*배당 암시확률"
    )

# ----------------------------
# 경기 목록
# ----------------------------
fixtures = []
if fetch_btn:
    try:
        fixtures = af_list_fixtures(league_id, int(season), the_date.strftime("%Y-%m-%d"))
        if not fixtures:
            st.warning("이 날짜엔 선택한 리그의 경기가 없어요.")
    except Exception as e:
        st.error(f"경기 불러오기 실패: {e}")

fixture = None
if fixtures:
    options = []
    for f in fixtures:
        home = f["teams"]["home"]["name"]
        away = f["teams"]["away"]["name"]
        kick = f["fixture"]["date"]
        try:
            kick_short = parse_dt(kick).strftime("%H:%M")
        except Exception:
            kick_short = kick
        options.append((f"{kick_short}  {home} vs {away}", f))
    label = st.selectbox("경기를 선택하세요", options, format_func=lambda x: x[0])[0]
    for text, obj in options:
        if text == label:
            fixture = obj
            break

# ----------------------------
# 선택된 경기 상세 + 피처 구축
# ----------------------------
if fixture:
    hid = fixture["teams"]["home"]["id"]
    aid = fixture["teams"]["away"]["id"]
    hname = fixture["teams"]["home"]["name"]
    aname = fixture["teams"]["away"]["name"]
    fid = fixture["fixture"]["id"]

    # 자동 새로고침
    auto = st.toggle("실시간 업데이트(자동 새로고침)", value=False, help=f"경기 중 켜두면 {AUTO_REFRESH_SEC}s마다 최신 이벤트 반영")
    if auto:
        st.autorefresh(interval=AUTO_REFRESH_SEC * 1000, key="auto_refresh_key", limit=None)

    st.subheader(f"경기: {hname} (홈) vs {aname} (원정)")

    # 버튼/입력 UI
    col_top = st.columns(5)
    with col_top[0]:
        load_btn = st.button("라인업/이벤트 불러오기", type="primary")
    with col_top[1]:
        odds_btn = st.button("배당 불러오기")
    with col_top[2]:
        injured_home = st.number_input("결장 주전(홈)", min_value=0, max_value=11, value=0, step=1)
    with col_top[3]:
        odds_slider = st.slider("배당 신호(슬라이더)", -1.0, 1.0, 0.0, 0.05)
    with col_top[4]:
        show_scores = st.toggle("xG/스코어 분포 보기", value=True)

    # 기본값
    form_diff = 0.0
    attack_diff = 0.0
    defense_diff = 0.0
    home_form_str = away_form_str = "?"

    # ----- 라인업/이벤트 -----
    if load_btn or auto:
        try:
            lus = af_lineups(fid)
            evs = af_events(fid)

            lu_home = lu_away = None
            for lu in lus:
                if lu["team"]["id"] == hid: lu_home = lu
                elif lu["team"]["id"] == aid: lu_away = lu

            # 라인업 기반 atk/dfn
            atk_h, dfn_h = calc_team_ratings(lu_home)
            atk_a, dfn_a = calc_team_ratings(lu_away)
            attack_diff = atk_h - atk_a
            defense_diff = dfn_h - dfn_a

            # 포메이션 매칭 가중치
            home_form_str = (lu_home or {}).get("formation") or "?"
            away_form_str = (lu_away or {}).get("formation") or "?"
            delta = FORMATION_MATCHUPS.get((home_form_str, away_form_str), (0.0, 0.0))
            attack_diff += delta[0]
            defense_diff += delta[1]

            # 선수 포지션 인덱스(교체용)
            home_pos_idx = build_player_bucket_index(lu_home)
            away_pos_idx = build_player_bucket_index(lu_away)

            # 이벤트 반영(시간가중)
            goals_h = goals_a = 0
            red_h = red_a = 0
            d_atk_h = d_atk_a = 0.0
            d_dfn_h = d_dfn_a = 0.0
            for e in evs:
                t_id = e.get("team", {}).get("id")
                etype = (e.get("type") or "").lower()
                edetail = (e.get("detail") or "").lower()
                w = minute_remaining_weight(e)

                # 득점
                if etype == "goal":
                    if t_id == hid:
                        goals_h += 1; d_atk_h += 0.25*w; d_dfn_h += 0.12*w
                    else:
                        goals_a += 1; d_atk_a += 0.25*w; d_dfn_a += 0.12*w

                # 카드
                if etype == "card":
                    if "red" in edetail:
                        if t_id == hid:
                            red_h += 1; d_atk_h -= 0.20*w; d_dfn_h -= 0.60*w
                        else:
                            red_a += 1; d_atk_a -= 0.20*w; d_dfn_a -= 0.60*w
                    elif "yellow" in edetail:
                        if t_id == hid:
                            d_atk_h -= 0.04*w; d_dfn_h -= 0.04*w
                        else:
                            d_atk_a -= 0.04*w; d_dfn_a -= 0.04*w

                # 교체
                if etype == "subst":
                    in_pl = (e.get("assist") or {}).get("id")
                    out_pl = (e.get("player") or {}).get("id")
                    if t_id == hid:
                        in_b = home_pos_idx.get(in_pl, "UNK"); out_b = home_pos_idx.get(out_pl, "UNK")
                        da, dd = bucket_weights_delta(in_b, out_b)
                        d_atk_h += da*w; d_dfn_h += dd*w
                    elif t_id == aid:
                        in_b = away_pos_idx.get(in_pl, "UNK"); out_b = away_pos_idx.get(out_pl, "UNK")
                        da, dd = bucket_weights_delta(in_b, out_b)
                        d_atk_a += da*w; d_dfn_a += dd*w

                # 부상
                if "injur" in edetail:
                    if t_id == hid:
                        d_atk_h -= 0.10*w; d_dfn_h -= 0.10*w
                    else:
                        d_atk_a -= 0.10*w; d_dfn_a -= 0.10*w

            # 합산
            attack_diff += 0.25*(goals_h - goals_a) + (d_atk_h - d_atk_a)
            defense_diff += 0.15*(goals_h - goals_a) + (d_dfn_h - d_dfn_a)

            if load_btn:
                st.success(f"라인업 {len(lus)}개, 이벤트 {len(evs)}개 불러옴 · 포메이션 {home_form_str} vs {away_form_str}")
                with st.expander("원시 라인업/이벤트 일부 보기"):
                    st.json({"lineups_sample": lus[:1], "events_sample": evs[:5]})
        except Exception as e:
            st.error(f"라인업/이벤트 불러오기 실패: {e}")

    # ----- 배당 → 암시확률/odds_signal/β_hat -----
    if odds_btn or auto:
        try:
            odds_resp = af_odds_by_fixture(fid)
            home_p = draw_p = away_p = None
            for entry in odds_resp:
                for b in entry.get("bookmakers", []):
                    for market in b.get("bets", []):
                        if (market.get("name") or "").lower() in ("match winner", "1x2", "winner"):
                            for val in market.get("values", []):
                                nm = (val.get("value") or "").upper()
                                odd = float(val.get("odd", 0) or 0)
                                if odd > 0:
                                    p = 1.0 / odd
                                    if "HOME" in nm or nm == "1": home_p = p
                                    elif "DRAW" in nm or nm == "X": draw_p = p
                                    elif "AWAY" in nm or nm == "2": away_p = p
                            break
                    if home_p and draw_p and away_p: break
                if home_p and draw_p and away_p: break
            if home_p and draw_p and away_p:
                s = home_p + draw_p + away_p
                home_p, draw_p, away_p = home_p/s, draw_p/s, away_p/s
                st.session_state._odds_prob_home = home_p

                # 사전 모델(odds_signal 0)과 로짓 차이로 β_odds 추정
                pre_feats = {"home_adv": 1, "form_diff": 0.0, "injured_starters": 0,
                             "attack_rating_diff": 0.0, "defense_rating_diff": 0.0, "odds_signal": 0.0}
                pre_home = win_prob_with_beta(pre_feats, BETA["odds_signal"])
                delta_logit = logit(home_p) - logit(pre_home)
                x = st.session_state.get("_odds_signal", 0.0)
                if abs(x) > 1e-6:
                    w_hat = delta_logit / x
                    prev = st.session_state.get("_beta_odds_hat", BETA["odds_signal"])
                    st.session_state._beta_odds_hat = 0.9*prev + 0.1*float(w_hat)

                st.session_state._odds_signal = round(home_p - pre_home, 3)
                if odds_btn:
                    st.success(f"암시확률(H/D/A): {home_p:.3f}/{draw_p:.3f}/{away_p:.3f} → odds_signal={st.session_state._odds_signal:+.3f}")
            elif odds_btn:
                st.warning("배당을 찾지 못했어요(마켓/북메이커 부재).")
        except Exception as e:
            st.error(f"배당 불러오기 실패: {e}")

    # ----- 최근 5경기 폼 (H/A 분리) -----
    split = st.toggle("홈/원정 분리 폼 사용", value=True, help="홈팀=홈경기, 원정팀=원정경기만 집계")
    form_cols = st.columns([1, 1, 1])
    with form_cols[0]:
        if st.button("최근 5경기 폼 계산", use_container_width=True):
            try:
                last_home_all = af_team_last_fixtures(hid, last=11)
                last_away_all = af_team_last_fixtures(aid, last=11)
                if split:
                    home_only = [f for f in last_home_all if is_home_for_team(f, hid)]
                    away_only = [f for f in last_away_all if not is_home_for_team(f, aid)]
                    last_home = pick_last_completed(home_only, 5)
                    last_away = pick_last_completed(away_only, 5)
                else:
                    last_home = pick_last_completed(last_home_all, 5)
                    last_away = pick_last_completed(last_away_all, 5)
                ppg_home = points_per_game(last_home, hid)
                ppg_away = points_per_game(last_away, aid)
                form_diff_calc = round(ppg_home - ppg_away, 3)
                st.session_state._form_calc = {"ppg_home": ppg_home, "ppg_away": ppg_away, "form_diff": form_diff_calc, "split": split}
                tag = "H/A" if split else "전체"
                st.success(f"[{tag}] 최근5경기 승점/경기: 홈 {ppg_home} · 원정 {ppg_away} → 차이 {form_diff_calc:+.3f}")
            except Exception as e:
                st.error(f"폼 계산 실패: {e}")
    with form_cols[1]:
        if st.session_state.get("_form_calc"):
            lab = "폼 차이(자동·H/A)" if st.session_state["_form_calc"].get("split") else "폼 차이(자동·전체)"
            st.metric(lab, f"{st.session_state['_form_calc']['form_diff']:+.3f}")
    with form_cols[2]:
        form_diff_manual = st.number_input(
            "폼 차이 수동입력",
            value=float(st.session_state.get("_form_calc", {}).get("form_diff", 0.0)),
            step=0.05,
            format="%.2f",
        )

    # ----- 예측하기 + xG/포아송 -----
    if st.button("예측하기", type="secondary"):
        form_value = st.session_state.get("_form_calc", {}).get("form_diff", form_diff_manual)
        odds_value = st.session_state.get("_odds_signal", odds_slider)
        beta_odds_eff = st.session_state.get("_beta_odds_hat", BETA["odds_signal"])

        feats = {
            "home_adv": 1,
            "form_diff": form_value,
            "injured_starters": injured_home,
            "attack_rating_diff": attack_diff,
            "defense_rating_diff": defense_diff,
            "odds_signal": odds_value,
        }
        # 모델 확률
        p_model = win_prob_with_beta(feats, beta_odds_eff)
        # 배당 혼합
        odds_home = st.session_state.get("_odds_prob_home", None)
        if odds_home is not None:
            p_win = (1 - cal_alpha) * p_model + cal_alpha * odds_home
        else:
            p_win = p_model
        p_draw = 0.25 * (1 - p_win)
        p_lose = 1 - p_win - p_draw

        st.subheader("결과")
        c1, c2, c3 = st.columns(3)
        c1.metric("홈 승", f"{p_win*100:.1f}%")
        c2.metric("무승부", f"{p_draw*100:.1f}%")
        c3.metric("원정 승", f"{p_lose*100:.1f}%")
        st.caption(f"포메이션: {home_form_str} vs {away_form_str}")

        # --- xG/포아송 스코어 분포 ---
        if show_scores:
            # 홈/원정 피처로 λ 추정 (간이)
            h_feats = {"attack_rating_diff": attack_diff, "defense_rating_diff": defense_diff,
                       "form_diff": form_value, "home_adv": 1}
            a_feats = {"attack_rating_diff": -attack_diff, "defense_rating_diff": -defense_diff,
                       "form_diff": -form_value, "home_adv": 0}
            lam_h, lam_a = estimate_expected_goals(h_feats, a_feats, base_goals=AVG_GOALS)
            df_scores = score_matrix(lam_h, lam_a, gmax=6)
            ph, pd, pa, btts, over25, under25, top3 = derive_markets_from_score(df_scores)

            st.markdown(f"**추정 xG:** 홈 {lam_h:.2f} · 원정 {lam_a:.2f}  |  **BTTS:** {btts*100:.1f}%  |  **O/U2.5:** {over25*100:.1f}% / {under25*100:.1f}%")
            # 상위 스코어 3개
            st.write("가장 가능성 높은 스코어 Top 3:")
            st.table(top3.assign(prob=lambda d: (d["prob"]*100).round(2)).rename(columns={"home":"홈","away":"원정","prob":"확률(%)"}))

            # 히트맵
            pivot = df_scores.pivot(index="home", columns="away", values="prob").fillna(0)
            df_heat = pivot.reset_index().melt("home", var_name="away", value_name="prob")
            heat = (
                alt.Chart(df_heat)
                .mark_rect()
                .encode(
                    x=alt.X("away:O", title="원정 득점"),
                    y=alt.Y("home:O", title="홈 득점"),
                    tooltip=[alt.Tooltip("home:O"), alt.Tooltip("away:O"), alt.Tooltip("prob:Q", format=".3%")],
                    color=alt.Color("prob:Q", title="확률", scale=alt.Scale(scheme="blues")),
                )
                .properties(height=260)
            )
            st.altair_chart(heat, use_container_width=True)

        st.divider()
        st.subheader("기여도 (β×x)")
        contributions = {
            "홈이점": BETA["home_adv"] * feats["home_adv"],
            "최근폼차": BETA["form_diff"] * feats["form_diff"],
            "결장주전(홈)": BETA["injured_starters"] * feats["injured_starters"],
            "공격지표차": BETA["attack_rating_diff"] * feats["attack_rating_diff"],
            "수비지표차": BETA["defense_rating_diff"] * feats["defense_rating_diff"],
            "배당신호(β̂)": beta_odds_eff * feats["odds_signal"],
        }
        df_bar = pd.DataFrame([{"feature": k, "value": v} for k, v in contributions.items()])
        chart = (
            alt.Chart(df_bar)
            .mark_bar()
            .encode(
                x=alt.X("value:Q", title="기여도 (β×x)"),
                y=alt.Y("feature:N", sort="-x", title="특성"),
                tooltip=[alt.Tooltip("feature:N", title="특성"),
                         alt.Tooltip("value:Q", title="기여도", format=".3f")],
            ).properties(height=240)
        )
        rule = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(strokeDash=[4, 4]).encode(x="x:Q")
        st.altair_chart(chart + rule, use_container_width=True)

else:
    st.info("사이드바에서 리그/시즌/날짜를 고르고 **경기 불러오기 → 경기 선택 → (선택) 라인업/이벤트/배당 → 폼 계산(H/A) → 예측하기** 순서로 사용하세요.")
