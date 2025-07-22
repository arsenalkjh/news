import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import date
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# 뉴스 데이터 읽기
@st.cache_data
def load_data(path):
    df = pd.read_excel(path)
    df = df[df['통합 분류1']=="경제>부동산"]
    # 1. 각 행을 리스트로 변환 (키워드별로 ','로 분리)
    docs = df['키워드'].astype(str).tolist()
    docs = [doc.replace(' ', '').split(',') for doc in docs]

    # 2. 다시 문자열로 합쳐서 벡터화 (LDA는 텍스트 입력 필요)
    docs_joined = [' '.join(tokens) for tokens in docs]

    # 3. CountVectorizer로 단어 행렬 생성
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(docs_joined)

    # 4. LDA 모델 학습 (주제 7개)
    lda = LatentDirichletAllocation(n_components=7, random_state=42)
    lda.fit(X)

    # 5. 각 행(뉴스)에 대해 주제 할당
    topic_assignments = lda.transform(X).argmax(axis=1)
    df['주제'] = topic_assignments
    return df
df = load_data("news.xlsx")

st.title("부동산")

col1, col2, col3 = st.columns(3)

def render_card(col, df, 주제값, col_name):
    col.write(f"주제: {주제값+1}")
    # 주제값 필터링 후 상위 3개 선택
    filtered = df[df["주제"] == 주제값].head(3)
    
    for i, row in filtered.iterrows():
        if col.button(row["제목"], key=f"{col_name}_{i}"):
            col.success(f"✅ {col_name}: '{row['제목']}' 선택됨")

# 각각의 컬럼에 주제=0인 카드 생성
render_card(col1, df, 주제값=0, col_name="컬럼1")
render_card(col2, df, 주제값=1, col_name="컬럼2")
render_card(col3, df, 주제값=2, col_name="컬럼3")


# 그룹별 집계
result = df.groupby(["일자", "주제"]).count().reset_index()
result["일자"] = result["일자"].astype(str).apply(lambda x: f"{x[2:4]}/{x[4:6]}/{x[6:8]}")

# Plotly로 주제별 시계열 그래프 그리기 (y축: 제목 개수)
fig = px.line(
    result,
    x="일자",
    y="제목",
    color="주제",
    markers=True,
    title="주제별 뉴스 건수 시계열 (Plotly)"
)

# 각 주제별 평균선 추가
# for topic in result['주제'].unique():
#     mean_val = result[result['주제'] == topic]['제목'].mean()
#     fig.add_hline(
#         y=mean_val,
#         line_dash="dot",
#         line_color=px.colors.qualitative.Plotly[topic % 10],
#         annotation_text=f"주제 {topic} 평균",
#         annotation_position="top left",
#         row="all", col="all"
#     )

fig.update_layout(
    xaxis_title="일자",
    yaxis_title="제목 개수",
    width=1000,
    height=400
)
st.plotly_chart(fig)

st.button("버튼")


# 기본 날짜 설정
start_date = date(2024, 1, 1)
end_date = date(2024, 12, 31)

# 사이드바 캘린더 (기간 선택)
selected_dates = st.sidebar.date_input(
    "기간 선택",
    [start_date, end_date]
)

# 선택 결과 표시
if len(selected_dates) == 2:
    st.write(f"선택한 기간: {selected_dates[0]} ~ {selected_dates[1]}")
else:
    st.write(f"선택한 날짜: {selected_dates}")
