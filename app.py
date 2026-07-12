import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import numpy as np
import altair as alt
import os
import urllib.request
import cv2
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from google import genai

# 1. 페이지 레이아웃 세팅 (넓은 화면으로 좌우 분할)
st.set_page_config(page_title="AI 표정 인식 & 멘토링 챗봇", layout="wide")
st.title("🔮 AI 실시간 표정 인식 & Gemini 피드백 시스템")
st.write("캠을 켜면 AI가 표정을 분석하고, 우측의 제미나이 챗봇이 당신의 감정에 맞는 피드백을 줍니다.")

# 2. 안전하게 인공지능 두뇌(.pth) 다운로드 및 로드
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

# 이미지 전처리 파이프라인
img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 3. Streamlit Secrets 금고에서 안전하게 구글 API 키 가져오기
try:
    gemini_key = st.secrets["GEMINI_API_KEY"]
    ai_client = genai.Client(api_key=gemini_key)
except Exception:
    st.error("🔑 Streamlit Secrets에 'GEMINI_API_KEY'가 설정되지 않았습니다. 설정을 확인해 주세요.")
    st.stop()

# 세션 상태(변수 저장소) 초기화
if "current_emotion" not in st.session_state:
    st.session_state.current_emotion = "neutral"
if "chatbot_response" not in st.session_state:
    st.session_state.chatbot_response = "카메라를 켜면 당신의 표정을 분석하여 챗봇이 다정한 피드백을 시작합니다!"

# 4. 실시간 웹캠 비디오 프레임 처리 클래스 정의
class EmotionTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # OpenCV를 이용해 실시간 화면을 RGB로 변환 후 모델 예측
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        img_tensor = img_transform(pil_img).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = F.softmax(outputs, dim=1).numpy()[0]
            
        max_idx = np.argmax(probabilities)
        pred_emotion = classes[max_idx]
        
        # 감정이 바뀌었을 때만 세션 상태를 업데이트하여 제미나이 호출 준비
        if st.session_state.current_emotion != pred_emotion:
            st.session_state.current_emotion = pred_emotion
            
        # 화면에 예측된 감정 텍스트 띄워주기 (OpenCV 서포트)
        cv2.putText(img, f"EMOTION: {pred_emotion.upper()}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return img

# 5. 화면 레이아웃 분할 (좌측: 캠/그래프, 우측: 제미나이 챗봇 피드백)
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🎥 실시간 웹캠 입력")
    # 웹캠 스트리머 가동
    webrtc_streamer(key="emotion-streamer", video_transformer_factory=EmotionTransformer)
    
    st.write(f"📊 현재 감지된 감정: **{st.session_state.current_emotion.upper()}**")
    
    # 임의의 시각화 피드백용 (실시간 변동 가이드)
    st.caption("팁: 캠 화면 속 표정이 바뀌면 하단의 피드백 요청 버튼을 눌러보세요!")

with col2:
    st.subheader("🤖 Gemini 감정 케어 멘토")
    
    # 사용자가 피드백 버튼을 누르면 제미나이 API 호출
    if st.button("🔄 현재 내 표정으로 피드백 받기"):
        with st.spinner("💭 제미나이가 당신의 표정을 분석하여 답변을 생각하고 있습니다..."):
            try:
                prompt = f"""
                사용자의 현재 실시간 표정 분석 결과 감정 상태는 [{st.session_state.current_emotion}] 입니다.
                이 감정 상태에 맞는 다정한 위로, 공감, 혹은 상황에 맞는 긍정적인 피드백을 친구처럼 친근한 말투로 딱 2~3문장 이내로 해줘.
                말끝에는 감정에 어울리는 이모지도 섞어줘.
                """
                response = ai_client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=prompt
                )
                st.session_state.chatbot_response = response.text
            except Exception as e:
                st.session_state.chatbot_response = f"구글 API 호출 중 오류가 발생했습니다: {e}"

    # 챗봇 스타일 말풍선 디자인 출력
    st.info(st.session_state.current_emotion.upper() + " 상태에 대한 멘토의 조언:")
    st.chat_message("assistant").write(st.session_state.chatbot_response)
