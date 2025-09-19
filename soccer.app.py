import math
import os
import requests
import pandas as pd
import altair as alt
import streamlit as st
from datetime import date
from dateutil.parser import parse as parse_dt

st.set_page_config(page_title="축구 승률 예측 – API-FOOTBALL", layout="centered")
st.title("⚽ 승률 예측 – API-FOOTBALL 연동")
st.caption("리그/날짜별 경기 조회 → 라인업/이벤트 반영 → 승률 예측 (MVP)")
APP_VERSION = "v0.2 (API-FOOTBALL)"
st.caption(f"현재 파일: {__file__} • 버전: {APP_VERSION}")


# =========================
# 0) 설정/키
# =========================
API_FOOTBALL_KEY = st.secrets.get("APIFOOTBALL_KEY") or os.getenv("APIFOOTBALL_KEY")
if not API_FOOTBALL_KEY:
    st.warning("`.streamlit/secrets.toml`에 APIFOOTBALL_KEY를 넣어주세요.")
BASE = "https://v3.football.api-sports.io"

LEAGUES = {
    "Premier League": 39,   # 잉글랜드
    "La Liga": 140,
    "Serie A": 135,
    "Bundesliga": 78,
    "Ligue 1": 61,
}

BETA = {  # 임시 가중치(튜닝/학습 예정)
    "bias": 0.0,
    "home_adv": 0.35,
    "form_diff": 0.80,
    "injured_starters": -0.40,
    "attack_rating_diff": 1.10,
    "defense_rating_diff": 0.70,
    "odds_signal": 1.20,
}

def sigmoid(z): 
    return 1/(1+math.exp(-z))

def win_prob(features: dict) -> float:
    z = (BETA["bias"]
         + BETA["home_adv"] * features.get("home_adv", 0)
         + BETA["form_diff"] * features.get("form_diff", 0)
         + BETA["injured_starters"] * features.get("injured_starters", 0)
         + BETA["attack_rating_diff"] * features.get("attack_rating_diff", 0)
         + BETA["defense_rating_diff"] * features.get("defense_rating_diff", 0)
         + BETA["odds_signal"] * features.get("odds_signal", 0))
    return sigmoid(z)

# =========================
# 1) API-FOOTBALL 헬퍼
# =========================
def _headers():
    return {"x-apisports-key": API_FOOTBALL_KEY}

@st.cache_data(ttl=1800)  # 30분 캐시
def af_get(path, params=None):
    r = requests.get(BASE + path, headers=_headers(), params=params, timeout=20)
    r.raise_for_status()
    data = r.json()
    if data.get("errors"):
        # {'rateLimit': ...} 같은 메시지가 들어올 수 있음
        raise RuntimeError(str(data["errors"]))
    return data.get("response", [])

@st.cache_data(ttl=3600)
def af_list_fixtures(league_id:int, season:int, the_date:str):
    """YYYY-MM-DD 날짜의 리그 경기 목록"""
    return af_get("/fixtures", {"league": league_id, "season": season, "date": the_date})

@st.cache_data(ttl=600)
def af_lineups(fixture_id:int):
    return af_get("/fixtures/lineups", {"fixture": fixture_id})

@st.cache_data(ttl=60)
def af_events(fixture_id:int):
    return af_get("/fixtures/events", {"fixture": fixture_id})

# =========================
# 2) 사이드바: 리그/시즌/날짜 선택
# =========================
with st.sidebar:
    st.header("⚙️ 데이터 선택")
    season = st.number_input("시즌(연도)", min_value=2015, max_value=2030, value=date.today().year, step=1)
    league_name = st.selectbox("리그", list(LEAGUES.keys()), index=0)
    league_id = LEAGUES[league_name]
    the_date = st.date_input("날짜", value=date.today())
    fetch_btn = st.button("경기 불러오기", use_container_width=True)

# =========================
# 3) 경기 리스트 로드
# =========================
fixtures = []
if fetch_btn:
    try:
        fixtures = af_list_fixtures(league_id, int(season), the_date.strftime("%Y-%m-%d"))
        if not fixtures:
            st.warning("이 날짜엔 선택한 리그의 경기가 없어요.")
    except Exception as e:
        st.error(f"경기 불러오기 실패: {e}")

# =========================
# 4) 경기 선택 UI
# =========================
fixture = None
if fixtures:
    # 보기 좋은 텍스트 구성
    options = []
    for f in fixtures:
        home = f["teams"]["home"]["name"]
        away = f["teams"]["away"]["name"]
        kick = f["fixture"]["date"]  # ISO
        try:
            kick_short = parse_dt(kick).strftime("%H:%M")
        except Exception:
            kick_short = kick
        options.append((f'{kick_short}  {home} vs {away}', f))

    label = st.selectbox("경기를 선택하세요", options, format_func=lambda x: x[0])[0]
    # 선택된 라벨로 fixture 결정
    for text, obj in options:
        if text == label:
            fixture = obj
            break

# =========================
# 5) 라인업/이벤트 로드 + 피처 산출
# =========================
if fixture:
    hid = fixture["teams"]["home"]["id"]
    aid = fixture["teams"]["away"]["id"]
    hname = fixture["teams"]["home"]["name"]
    aname = fixture["teams"]["away"]["name"]
    fid = fixture["fixture"]["id"]

    st.subheader(f"경기: {hname} (홈) vs {aname} (원정)")

    col = st.columns(3)
    with col[0]:
        load_btn = st.button("라인업/이벤트 불러오기", type="primary")
    with col[1]:
        injured_home = st.number_input("결장 주전 수(홈)", min_value=0, max_value=11, value=0, step=1)
    with col[2]:
        odds_signal = st.slider("배당 신호(선택)", -1.0, 1.0, 0.0, 0.05)

    # 기본 자동값(라인업 없을 때 대비)
    form_diff = 0.0
    attack_diff = 0.0
    defense_diff = 0.0

    if load_btn:
        try:
            lus = af_lineups(fid)     # 선발/포메이션
            evs = af_events(fid)      # 골/카드/교체

            # --- 5-1) 라인업 → 공/수 레이팅 간이 계산 ---
            # 아이디어: 포지션/역할 기준으로 가중합
            # ATT: FW/W/AM / MID: CM/DM / DEF: CB/FB
            def calc_team_ratings(lineup_obj):
                if not lineup_obj:
                    return 0.0, 0.0
                start = lineup_obj.get("startXI", [])
                atk, dfn = 0.0, 0.0
                for p in start:
                    pos = (p["player"].get("pos") or "").upper()  # 일부 응답은 'pos' 대신 'grid'만 있을 수 있음
                    # 대략적 분류 (간이)
                    if any(k in pos for k in ["FW", "ST", "LW", "RW", "AM"]):
                        atk += 1.0
                    elif any(k in pos for k in ["CM", "MF", "WM", "DM"]):
                        atk += 0.4; dfn += 0.4
                    elif any(k in pos for k in ["CB", "LB", "RB", "DF"]):
                        dfn += 1.0
                    else:
                        # 포지션 정보 빈약하면 중립값
                        atk += 0.3; dfn += 0.3
                # 선발 11명 기준 정규화
                return atk/11.0, dfn/11.0

            # 홈/원정 라인업 추출
            lu_home = None; lu_away = None
            for lu in lus:
                if lu["team"]["id"] == hid:
                    lu_home = lu
                elif lu["team"]["id"] == aid:
                    lu_away = lu

            atk_h, dfn_h = calc_team_ratings(lu_home)
            atk_a, dfn_a = calc_team_ratings(lu_away)

            attack_diff = (atk_h - atk_a)   # 홈공격 - 원정공격
            defense_diff = (dfn_h - dfn_a)  # 홈수비 - 원정수비

            # --- 5-2) 이벤트 → 득점/레드카드 반영(간이 룰) ---
            goals_h = goals_a = 0
            red_h = red_a = 0
            for e in evs:
                t_id = e["team"]["id"]
                etype = (e.get("type") or "").lower()
                edetail = (e.get("detail") or "").lower()
                if etype == "goal":
                    if t_id == hid: goals_h += 1
                    else:           goals_a += 1
                if etype == "card" and "red" in edetail:
                    if t_id == hid: red_h += 1
                    else:           red_a += 1

            # 골 차이를 공격/수비 차이에 약간 반영(예: 득점하면 공격↑, 실점하면 수비↓)
            attack_diff += 0.25 * (goals_h - goals_a)
            defense_diff += 0.15 * ((goals_h - goals_a))  # 리드 시 수비적으로도 안정

            # 레드카드 패널티(남은 시간 고려 못 했으니 간이 고정치)
            defense_diff += (-0.25 * red_h) + (0.25 * red_a)

            # 최근폼 차이는 아직 API만으로 간단 추정 어려우니 0으로 두고,
            # 나중에 팀 최신 경기 결과 평균을 불러와서 승점/경기 차이로 계산 예정
            form_diff = 0.0

            st.success(f"라인업 {len(lus)}개, 이벤트 {len(evs)}개 불러옴")
            with st.expander("원시 라인업/이벤트 확인"):
                st.json({"lineups": lus[:1], "events_sample": evs[:5]})

        except Exception as e:
            st.error(f"라인업/이벤트 불러오기 실패: {e}")

    # ---- 5-3) 예측하기 ----
    if st.button("예측하기", type="secondary"):
        feats = {
            "home_adv": 1,
            "form_diff": form_diff,
            "injured_starters": injured_home,
            "attack_rating_diff": attack_diff,
            "defense_rating_diff": defense_diff,
            "odds_signal": odds_signal,
        }
        p_win = win_prob(feats)
        p_draw = 0.25 * (1 - p_win)
        p_lose = 1 - p_win - p_draw

        st.subheader("결과")
        c1, c2, c3 = st.columns(3)
        c1.metric("홈 승", f"{p_win*100:.1f}%")
        c2.metric("무승부", f"{p_draw*100:.1f}%")
        c3.metric("원정 승", f"{p_lose*100:.1f}%")

        st.divider()
        st.subheader("기여도 (β×x)")
        contributions = {
            "홈이점": BETA["home_adv"] * feats["home_adv"],
            "최근폼차": BETA["form_diff"] * feats["form_diff"],
            "결장주전(홈)": BETA["injured_starters"] * feats["injured_starters"],
            "공격지표차": BETA["attack_rating_diff"] * feats["attack_rating_diff"],
            "수비지표차": BETA["defense_rating_diff"] * feats["defense_rating_diff"],
            "배당신호": BETA["odds_signal"] * feats["odds_signal"],
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
        rule = alt.Chart(pd.DataFrame({"x":[0]})).mark_rule(strokeDash=[4,4]).encode(x="x:Q")
        st.altair_chart(chart + rule, use_container_width=True)

else:
    st.info("사이드바에서 리그/시즌/날짜를 고르고 **경기 불러오기 → 경기 선택 → 라인업/이벤트 불러오기 → 예측하기** 순서로 진행하세요.")
