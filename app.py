import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import altair as alt
import os

# 1. 페이지 제목 및 레이아웃 설정
st.set_page_config(page_title="AI 표정 감정 인식기", layout="centered")
st.title("🔮 7개 표정 기반 AI 감정 분석기")
st.write("당신의 사진을 업로드하면 AI가 실시간으로 감정을 분석합니다.")

# 2. 모델 정의 및 고속 깃허브 서버에서 두뇌 파일(.pth) 자동 불러오기
@st.cache_resource
def load_emotion_model():
    model_path = 'emotion_resnet18.pth'
    
    # 서버에 모델 파일이 없으면 방금 만든 무적의 깃허브 릴리즈 주소에서 즉시 다운로드
    if not os.path.exists(model_path):
        with st.spinner("🚀 AI 두뇌(모델 파일)를 안전하게 로드하는 중입니다. 최초 1회만 진행됩니다..."):
            # ⚠️ [필수] 바로 위 2단계에서 복사한 본인의 릴리즈 주소를 아래 따옴표 안에 붙여넣으세요!
            download_url = 'https://github.com/3yearscrong1/face-emotion-ai/releases/tag/v1.0'
            
            import urllib.request
            opener = urllib.request.build_opener()
            opener.addheaders = [('User-agent', 'Mozilla/5.0')]
            urllib.request.install_opener(opener)
            urllib.request.urlretrieve(download_url, model_path)
    # 코랩에서 썼던 ResNet-18 구조 그대로 가중치 입히기
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, 7)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

try:
    model = load_emotion_model()
    classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
except Exception as e:
    st.error(f"모델 파일을 읽는 도중 오류가 발생했습니다: {e}")
    st.stop()

# 3. 이미지 업로드 UI 만들기
uploaded_file = st.file_uploader("얼굴 사진을 업로드하세요 (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='업로드된 이미지', width=350)
    
    # 4. 이미지 전처리 적용
    base_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_tensor = base_transform(image).unsqueeze(0)
    
    # 5. AI 예측 진행
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = F.softmax(outputs, dim=1).numpy()[0]
        
    max_idx = np.argmax(probabilities)
    predicted_emotion = classes[max_idx]
    confidence = probabilities[max_idx] * 100
    
    # 6. 결과 시각화 및 출력
    st.success(f"📊 분석 결과: 이 사람은 현재 **{confidence:.2f}%**의 확률로 **[{predicted_emotion.upper()}]** 상태입니다.")
    
    chart_data = {"Emotion": classes, "Probability (%)": probabilities * 100}
    chart = alt.Chart(alt.Data(values=[dict(zip(chart_data.keys(), v)) for v in zip(*chart_data.values())])).mark_bar().encode(
        x='Probability (%):Q',
        y=alt.Y('Emotion:N', sort='-x'),
        color=alt.value('skyblue')
    )
    st.altair_chart(chart, use_container_width=True)
