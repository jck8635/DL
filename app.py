import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn
import plotly.express as px
import plotly.graph_objects as go

# =========================================================
# 1. 페이지 설정
# =========================================================
st.set_page_config(
    page_title="유해 표현 탐지 인사이트 대시보드",
    page_icon="🛡️",
    layout="wide"
)

st.title("🛡️ 유해 표현 탐지 & 분석 대시보드")

MODEL_DIR = Path("models")
CHAR_MODEL_PATH = MODEL_DIR / "char_cnn_best.pt"
TEXT_MODEL_PATH = MODEL_DIR / "text_cnn_best.pt"
CHAR_VOCAB_PATH = MODEL_DIR / "char_vocab.pkl"
WORD_VOCAB_PATH = MODEL_DIR / "word_vocab.pkl"
CONFIG_PATH = MODEL_DIR / "model_config.pkl"

# 기본 설정 (config가 없을 경우)
MAX_CHAR_LENGTH = 300
MAX_WORD_LENGTH = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# 2. 모델 정의 (학습 코드와 동일)
# =========================================================
class CharCNN(nn.Module):
    def __init__(self, vocab_size, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 64, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(64, 128, kernel_size=k) for k in [3, 4, 5]
        ])
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128 * 3, num_classes)

    def forward(self, x):
        x = self.embedding(x).transpose(1, 2)
        features = []
        for conv in self.convs:
            out = torch.relu(conv(x))
            out = torch.max_pool1d(out, out.size(2)).squeeze(2)
            features.append(out)
        x = self.dropout(torch.cat(features, dim=1))
        return self.fc(x)


class TextCNN(nn.Module):
    def __init__(self, vocab_size, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 128, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(128, 128, kernel_size=k) for k in [2, 3, 4]
        ])
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128 * 3, num_classes)

    def forward(self, x):
        x = self.embedding(x).transpose(1, 2)
        features = []
        for conv in self.convs:
            out = torch.relu(conv(x))
            out = torch.max_pool1d(out, out.size(2)).squeeze(2)
            features.append(out)
        x = self.dropout(torch.cat(features, dim=1))
        return self.fc(x)


# =========================================================
# 3. 전처리 함수 (학습 코드와 동일)
# =========================================================
def tokenize_words(text: str):
    return re.findall(r"[가-힣]+|[a-zA-Z]+|\d+|[^\s]", str(text).lower())


def encode_text(text, vocab, max_length, mode="word"):
    tokens = list(str(text)) if mode == "char" else tokenize_words(text)
    ids = [vocab.get(token, vocab["<UNK>"]) for token in tokens[:max_length]]
    ids += [vocab["<PAD>"]] * (max_length - len(ids))
    return ids


# =========================================================
# 4. 모델 / 어휘 로드
# =========================================================
@st.cache_resource
def load_resources():
    # vocab
    if not CHAR_VOCAB_PATH.exists() or not WORD_VOCAB_PATH.exists():
        raise FileNotFoundError(
            f"어휘 사전 파일이 없습니다.\n"
            f"필요 파일: {CHAR_VOCAB_PATH}, {WORD_VOCAB_PATH}"
        )
    char_vocab = joblib.load(CHAR_VOCAB_PATH)
    word_vocab = joblib.load(WORD_VOCAB_PATH)

    # config (선택)
    max_char = MAX_CHAR_LENGTH
    max_word = MAX_WORD_LENGTH
    if CONFIG_PATH.exists():
        config = joblib.load(CONFIG_PATH)
        max_char = config.get("max_char_length", MAX_CHAR_LENGTH)
        max_word = config.get("max_word_length", MAX_WORD_LENGTH)

    # Char-CNN
    if not CHAR_MODEL_PATH.exists():
        raise FileNotFoundError(f"Char-CNN 모델이 없습니다: {CHAR_MODEL_PATH}")
    char_ckpt = torch.load(CHAR_MODEL_PATH, map_location=DEVICE)
    char_model = CharCNN(len(char_vocab)).to(DEVICE)
    char_model.load_state_dict(char_ckpt["model_state_dict"])
    char_model.eval()

    # TextCNN
    if not TEXT_MODEL_PATH.exists():
        raise FileNotFoundError(f"TextCNN 모델이 없습니다: {TEXT_MODEL_PATH}")
    text_ckpt = torch.load(TEXT_MODEL_PATH, map_location=DEVICE)
    text_model = TextCNN(len(word_vocab)).to(DEVICE)
    text_model.load_state_dict(text_ckpt["model_state_dict"])
    text_model.eval()

    return (
        char_model, text_model,
        char_vocab, word_vocab,
        max_char, max_word,
        char_ckpt.get("best_val_f1"), text_ckpt.get("best_val_f1"),
        char_ckpt.get("best_epoch"), text_ckpt.get("best_epoch")
    )


try:
    (
        char_model, text_model,
        char_vocab, word_vocab,
        max_char_len, max_word_len,
        char_best_f1, text_best_f1,
        char_best_epoch, text_best_epoch
    ) = load_resources()
except Exception as e:
    st.error(f"모델 또는 어휘 사전을 불러올 수 없습니다.\n\n{e}")
    st.info(
        "다음 파일들이 `models/` 폴더에 있는지 확인해주세요.\n"
        "- char_cnn_best.pt\n"
        "- text_cnn_best.pt\n"
        "- char_vocab.pkl\n"
        "- word_vocab.pkl\n"
        "- model_config.pkl (선택)"
    )
    st.stop()


def predict(text: str):
    """두 모델로 추론 후 결과 반환"""
    char_ids = encode_text(text, char_vocab, max_char_len, mode="char")
    word_ids = encode_text(text, word_vocab, max_word_len, mode="word")

    char_tensor = torch.tensor([char_ids], dtype=torch.long).to(DEVICE)
    word_tensor = torch.tensor([word_ids], dtype=torch.long).to(DEVICE)

    with torch.no_grad():
        char_logits = char_model(char_tensor)
        text_logits = text_model(word_tensor)

        char_probs = torch.softmax(char_logits, dim=1)[0].cpu().numpy()
        text_probs = torch.softmax(text_logits, dim=1)[0].cpu().numpy()

    char_pred = int(np.argmax(char_probs))
    text_pred = int(np.argmax(text_probs))

    is_harmful = (char_pred == 1)          # 유해 표현
    is_phishing = (text_pred == 1)         # 피싱·광고

    if is_harmful and is_phishing:
        final = "유해 표현 + 피싱·광고"
    elif is_harmful:
        final = "유해 표현"
    elif is_phishing:
        final = "피싱·광고"
    else:
        final = "정상"

    return {
        "final": final,
        "is_harmful": is_harmful,
        "is_phishing": is_phishing,
        "char_prob_harmful": float(char_probs[1]),
        "text_prob_phishing": float(text_probs[1]),
        "char_probs": char_probs,
        "text_probs": text_probs,
    }


# =========================================================
# 5. 탭 구성
# =========================================================
tab1, tab2, tab3 = st.tabs(["🔍 실시간 탐지", "📊 모델 성능 리포트", "💡 모델 정보"])

# ---------------------------------------------------------
# TAB 1: 실시간 분석
# ---------------------------------------------------------
with tab1:
    st.subheader("문장 실시간 유해성 분석")
    user_text = st.text_area(
        "분석할 문장을 입력하세요.",
        placeholder="예: 안녕하세요, 이벤트에 참여하세요!",
        height=120
    )

    if st.button("분석 실행", type="primary", use_container_width=True):
        if not user_text.strip():
            st.warning("문장을 입력해주세요.")
        else:
            result = predict(user_text)

            st.divider()

            # 최종 판정
            if result["final"] == "정상":
                st.success(f"✅ **정상:** 정상적인 문장으로 판단됩니다.")
            else:
                st.error(f"🚨 **주의:** {result['final']} 가능성이 높은 문장입니다.")

            # 확률 메트릭
            col1, col2 = st.columns(2)
            with col1:
                st.metric(
                    "Char-CNN (유해 표현) 확률",
                    f"{result['char_prob_harmful'] * 100:.2f}%"
                )
                st.progress(min(result["char_prob_harmful"], 1.0))
                st.caption("비속어 · 우회 유해 표현 탐지")
            with col2:
                st.metric(
                    "TextCNN (피싱·광고) 확률",
                    f"{result['text_prob_phishing'] * 100:.2f}%"
                )
                st.progress(min(result["text_prob_phishing"], 1.0))
                st.caption("스미싱 · 광고 · 피싱 탐지")

            # 상세 확률 막대 그래프
            st.markdown("#### 📊 클래스별 확률")
            prob_df = pd.DataFrame({
                "모델": ["Char-CNN", "Char-CNN", "TextCNN", "TextCNN"],
                "클래스": ["정상", "유해 표현", "정상", "피싱·광고"],
                "확률": [
                    result["char_probs"][0],
                    result["char_probs"][1],
                    result["text_probs"][0],
                    result["text_probs"][1],
                ]
            })
            fig = px.bar(
                prob_df,
                x="클래스",
                y="확률",
                color="모델",
                barmode="group",
                text_auto=".2%",
                title="모델별 클래스 확률",
                color_discrete_sequence=["#636EFA", "#EF553B"]
            )
            fig.update_layout(yaxis_tickformat=".0%", yaxis_range=[0, 1])
            st.plotly_chart(fig, use_container_width=True)

            # 요약 테이블
            summary = pd.DataFrame([
                {
                    "모델": "Char-CNN",
                    "탐지 대상": "비속어·유해 표현",
                    "판정": "탐지" if result["is_harmful"] else "미탐지",
                    "탐지 확률(%)": round(result["char_prob_harmful"] * 100, 2)
                },
                {
                    "모델": "TextCNN",
                    "탐지 대상": "피싱·광고",
                    "판정": "탐지" if result["is_phishing"] else "미탐지",
                    "탐지 확률(%)": round(result["text_prob_phishing"] * 100, 2)
                }
            ])
            st.dataframe(summary, use_container_width=True, hide_index=True)


# ---------------------------------------------------------
# TAB 2: 모델 성능 리포트
# ---------------------------------------------------------
with tab2:
    st.subheader("Char-CNN / TextCNN 모델 정보")
    st.info(
        "두 모델은 서로 다른 목적의 이진 분류기로 학습되었습니다.\n\n"
        "- **Char-CNN**: 정상(0) vs 유해 표현(1)\n"
        "- **TextCNN**: 정상(0) vs 피싱·광고(2 → 내부적으로 1)"
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Char-CNN 최적 Epoch", char_best_epoch if char_best_epoch else "-")
    m2.metric("Char-CNN 검증 F1", f"{char_best_f1:.4f}" if char_best_f1 else "-")
    m3.metric("TextCNN 최적 Epoch", text_best_epoch if text_best_epoch else "-")
    m4.metric("TextCNN 검증 F1", f"{text_best_f1:.4f}" if text_best_f1 else "-")

    st.markdown("---")
    st.markdown("### 모델 구조 요약")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Char-CNN**")
        st.code(
            "Embedding(64)\n"
            "Conv1d(kernel=3,4,5) → MaxPool\n"
            "Dropout(0.3)\n"
            "Linear → 2 classes",
            language="text"
        )
        st.caption(f"어휘 크기: {len(char_vocab):,} | Max Length: {max_char_len}")

    with col_b:
        st.markdown("**TextCNN**")
        st.code(
            "Embedding(128)\n"
            "Conv1d(kernel=2,3,4) → MaxPool\n"
            "Dropout(0.3)\n"
            "Linear → 2 classes",
            language="text"
        )
        st.caption(f"어휘 크기: {len(word_vocab):,} | Max Length: {max_word_len}")

    st.markdown("---")
    st.markdown("### 학습 데이터 구성 (참고)")
    dataset_df = pd.DataFrame({
        "Dataset": [
            "Dataset1 (욕설/비하)",
            "Dataset2 (우회 유해)",
            "Dataset3 (피싱/스팸)",
            "Dataset4 (피싱/광고)"
        ],
        "역할": [
            "Char-CNN (정상/유해)",
            "Char-CNN (정상/유해)",
            "TextCNN (정상/피싱)",
            "TextCNN (정상/피싱)"
        ]
    })
    st.dataframe(dataset_df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------
# TAB 3: 모델 정보 / 사용 가이드
# ---------------------------------------------------------
with tab3:
    st.subheader("모델 사용 가이드")

    st.markdown("""
    ### 이중 모델 탐지 로직
    | Char-CNN 결과 | TextCNN 결과 | 최종 판정          |
    |---------------|--------------|--------------------|
    | 유해 표현      | 피싱·광고     | 유해 표현 + 피싱·광고 |
    | 유해 표현      | 정상         | 유해 표현          |
    | 정상          | 피싱·광고     | 피싱·광고          |
    | 정상          | 정상         | 정상               |

    ### 전처리 방식
    - **Char-CNN**: 문자를 그대로 분리 (`list(str(text))`)
    - **TextCNN**: 정규식 기반 단어 토큰화  
      `[가-힣]+ | [a-zA-Z]+ | \\d+ | [^\\s]` (소문자 변환 포함)

    """)