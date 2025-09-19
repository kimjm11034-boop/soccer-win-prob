import math
import streamlit as st
import pandas as pd
import altair as alt

st.set_page_config(page_title="축구 승률 예측 – API-FOOTBALL", layout="centered")
st.title("⚽ 승률 예측 – API-FOOTBALL 연동")
st.caption("리그/날짜별 경기 조회 → 라인업/이벤트 반영 → 승률 예측 (MVP)")


# ---- 임시 가중치(나중에 데이터로 튜닝) ----
BETA = {
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

# ---- 입력 폼 ----
with st.form("inputs"):
    col1, col2 = st.columns(2)
    with col1:
        home = st.selectbox("홈/원정", ["홈", "원정"])
        form_diff = st.slider("최근폼 차이(승점/경기)", -1.0, 1.0, 0.0, 0.05)
        attack_diff = st.slider("선발 공격지표 차이", -1.0, 1.0, 0.0, 0.05)
    with col2:
        injured = st.number_input("결장 주전 수(우리팀)", min_value=0, max_value=11, value=0, step=1)
        defense_diff = st.slider("선발 수비지표 차이", -1.0, 1.0, 0.0, 0.05)
        odds_signal = st.slider("배당 신호(선택)", -1.0, 1.0, 0.0, 0.05)
    submitted = st.form_submit_button("예측하기")

# ---- 제출 시 결과 ----
if submitted:
    feats = {
        "home_adv": 1 if home == "홈" else 0,
        "form_diff": form_diff,
        "injured_starters": injured,
        "attack_rating_diff": attack_diff,
        "defense_rating_diff": defense_diff,
        "odds_signal": odds_signal,
    }
    p_win = win_prob(feats)
    # 간단 분배(입문용): 나중에 포아송/베이즈로 개선
    p_draw = 0.25 * (1 - p_win)
    p_lose = 1 - p_win - p_draw

    st.subheader("결과")
    st.metric("승리 확률", f"{p_win*100:.1f}%")
    st.write(f"무승부: {p_draw*100:.1f}% · 패배: {p_lose*100:.1f}%")

    st.divider()
    st.subheader("무엇이 확률을 움직였나? (β×x 크기)")
    contributions = {
        "홈이점": BETA["home_adv"] * feats["home_adv"],
        "최근폼차": BETA["form_diff"] * feats["form_diff"],
        "결장주전": BETA["injured_starters"] * feats["injured_starters"],
        "공격지표차": BETA["attack_rating_diff"] * feats["attack_rating_diff"],
        "수비지표차": BETA["defense_rating_diff"] * feats["defense_rating_diff"],
        "배당신호": BETA["odds_signal"] * feats["odds_signal"],
    }

    # ---- 가로 막대 (Altair) ----
    df = pd.DataFrame([{"feature": k, "value": v} for k, v in contributions.items()])
    st.caption("양수=유리, 음수=불리 (값이 클수록 승률에 더 큰 영향)")

    chart = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("value:Q", title="기여도 (β×x)"),
            y=alt.Y("feature:N", sort="-x", title="특성"),
            tooltip=[
                alt.Tooltip("feature:N", title="특성"),
                alt.Tooltip("value:Q", title="기여도", format=".3f"),
            ],
        )
        .properties(height=220)
    )
    rule = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(strokeDash=[4, 4]).encode(x="x:Q")

    st.altair_chart(chart + rule, use_container_width=True)

# ---- 제출 전 안내 ----
else:
    st.info("슬라이더/입력을 조정하고 **예측하기**를 눌러보세요!")
