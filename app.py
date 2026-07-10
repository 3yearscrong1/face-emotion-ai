import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import altair as alt
import os
import requests

# 1. 페이지 제목 및 레이아웃 설정
st.set_page_config(page_title="AI 표정 감정 인식기", layout="centered")
st.title("🔮 7개 표정 기반 AI 감정 분석기")
st.write("당신의 사진을 업로드하면 AI가 실시간으로 감정을 분석합니다.")

# 대용량 구글 드라이브 파일 다운로드 함수 정의
def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': id}, stream=True)
    token = None
    
    # 구글이 보내는 쿠키에서 대용량 경고 우회 토큰 찾기
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            token = value
            break

    # 우회 토큰이 있다면 주소에 실어서 재요청
    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)
        
    # 파일 저장하기
    with open(destination, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)

# 2. 모델 정의 및 구글 드라이브에서 두뇌 파일(.pth) 자동 불러오기
@st.cache_resource
def load_emotion_model():
    model_path = 'emotion_resnet18.pth'
    
    # 서버에 모델 파일이 없으면 구글 드라이브에서 우회 코드로 다운로드
    if not os.path.exists(model_path):
        with st.spinner("🚀 AI 두뇌(모델 파일)를 원격 서버에서 안전하게 로드하는 중입니다. 최초 1회만 진행됩니다..."):
            FILE_ID = '108oEl1sGNwfLW4nhViiSCelkRH1TGsiM'
            download_file_from_google_drive(FILE_ID, model_path)
            
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
    st.error("구글 드라이브 대용량 보안 필터링으로 인해 모델을 가져오지 못했습니다. 잠시 후 다시 시도해 주세요.")
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
