import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os
import urllib.request
import cv2
from google import genai

# 1. 페이지 레이아웃 세팅
st.set_page_config(page_title="AI 표정 인식 & 멘토링 챗봇", layout="wide")
st.title("🔮 AI 실시간 표정 인식 & Gemini 피드백 시스템")
st.write("카메라 화면에서 사진을 촬영하면, AI가 표정을 분석하고 제미나이 챗봇이 다정한 피드백을 줍니다.")

# 2. 안전하게 인공지능 두뇌(.pth) 로드
@st.cache_resource
def load_emotion_model():
    model_path = 'emotion_resnet18.pth'
    if not os.path.exists(model_path):
        with st.spinner("🚀 AI 가중치 파일을 다운로드 중입니다..."):
            download_url = 'https://github.com/3yearscrong1/face-emotion-ai/releases/download/v1.0/emotion_resnet18.pth'
            opener = urllib.request.build_opener()
            opener.addheaders = [('User-agent', 'Mozilla/5.0')]
            urllib.request.install_opener(opener)
            urllib.request.urlretrieve(download_url, model_path)
            
    model = models.resnet18()
    model.fc = nn.Linear(model.fc.in_features, 7)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=False))
    model.eval()
    return model

model = load_emotion_model()
classes = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 3. 구글 API 키 가져오기
try:
    gemini_key = st.secrets["GEMINI_API_KEY"]
    ai_client = genai.Client(api_key=gemini_key)
except Exception:
    st.error("🔑 Streamlit Secrets 설정을 확인해 주세요.")
    st.stop()

# 4. 화면 레이아웃 좌우 분할
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🎥 실시간 카메라 입력")
    # Streamlit 공식 내장 고화질 카메라 컴포넌트 가동 (버벅임, 멈춤 절대 없음)
    camera_image = st.camera_input("분석하고 싶은 표정을 짓고 [Take Photo]를 누르세요")

with col2:
    st.subheader("🤖 Gemini 감정 케어 멘토")
    
    # 사용자가 사진을 촬영(Take Photo)한 순간에만 내부 로직이 딱 1번 실행됨 (429 차단 원천 봉쇄)
    if camera_image is not None:
        with st.spinner("💭 AI가 표정을 분석하고 제미나이가 답변을 작성 중입니다..."):
            try:
                # 1) 촬영된 이미지를 PIL Image 및 가공 가능한 형태로 변환
                pil_img = Image.open(camera_image).convert('RGB')
                img_np = np.array(pil_img)
                
                # 2) [인식률 극대화] 명암 보정 필터(Equalization) 적용하여 HAPPY 등 감정선 뚜렷하게 강조
                gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
                equalized = cv2.equalizeHist(gray)
                processed_rgb = cv2.cvtColor(equalized, cv2.COLOR_GRAY2RGB)
                
                # 3) 딥러닝 모델 예측 진행
                final_pil = Image.fromarray(processed_rgb)
                img_tensor = img_transform(final_pil).unsqueeze(0)
                
                with torch.no_grad():
                    outputs = model(img_tensor)
                    probabilities = F.softmax(outputs, dim=1).numpy()[0]
                
                max_idx = np.argmax(probabilities)
                detected_emotion = classes[max_idx]
                confidence = probabilities[max_idx] * 100
                
                # 4) 감지 결과 레이블 출력
                st.success(f"📊 분석 결과: 현재 **{confidence:.2f}%**의 확률로 **[{detected_emotion.upper()}]** 상태입니다.")
                
                # 5) 고정된 감정 상태로 Gemini API 깔끔하게 1번 호출
                prompt = f"""
                사용자의 현재 표정 분석 감정 상태는 [{detected_emotion}] 입니다.
                이 감정에 맞는 따뜻한 위로, 공감, 혹은 응원의 피드백을 친구처럼 친근한 말투로 딱 2~3문장 이내로 작성해줘.
                말끝에는 감정에 어울리는 이모지(예시: 😊, 😭, ☕)를 자연스럽게 섞어줘.
                """
                response = ai_client.models.generate_content(
                    model='gemini-2.0-flash',
                    contents=prompt
                )
                
                # 6) 결과 출력
                st.info(f"{detected_emotion.upper()} 감정에 대한 멘토의 편지:")
                st.chat_message("assistant").write(response.text)
                
            except Exception as e:
                if "429" in str(e):
                    st.error("⚠️ 구글 무료 서버 제한에 도달했습니다. 5초 뒤에 사진을 다시 찍어보세요!")
                else:
                    st.error(f"연동 오류 발생: {e}")
    else:
        st.warning("📸 먼저 왼쪽 카메라 영역에서 [Take Photo] 버튼을 눌러 사진을 촬영해 주세요!")
