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
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
from google import genai
import queue
import time  # 💡 무분별한 연타 방지용 시간 라이브러리 추가

# 1. 페이지 레이아웃 세팅
st.set_page_config(page_title="AI 실시간 표정 인식 & 멘토링 챗봇", layout="wide")
st.title("🔮 AI 실시간 표정 인식 & Gemini 피드백 시스템")

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

# 3. 구글 API 키 및 클라이언트 초기화
try:
    gemini_key = st.secrets["GEMINI_API_KEY"]
    ai_client = genai.Client(api_key=gemini_key)
except Exception:
    st.error("🔑 Streamlit Secrets 설정을 확인해 주세요.")
    st.stop()

# 💡 [핵심] 연타 및 중복 호출 방지를 위한 세션 타임스탬프 초기화
if "chatbot_response" not in st.session_state:
    st.session_state.chatbot_response = "아래 버튼을 누르면 실시간 감정에 맞춘 피드백이 즉시 출력됩니다."
if "last_processed_emotion" not in st.session_state:
    st.session_state.last_processed_emotion = "NEUTRAL"
if "last_call_time" not in st.session_state:
    st.session_state.last_call_time = 0.0

# 4. 실시간 웹캠 비디오 프레임 처리 클래스
class EmotionProcessor(VideoProcessorBase):
    def __init__(self):
        self.result_queue = queue.Queue()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        equalized = cv2.equalizeHist(gray)
        img_rgb = cv2.cvtColor(equalized, cv2.COLOR_GRAY2RGB)
        
        small_img = cv2.resize(img_rgb, (224, 224))
        pil_img = Image.fromarray(small_img)
        img_tensor = img_transform(pil_img).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = F.softmax(outputs, dim=1).numpy()[0]
            
        max_idx = np.argmax(probabilities)
        pred_emotion = classes[max_idx]
        
        while not self.result_queue.empty():
            try:
                self.result_queue.get_nowait()
            except queue.Empty:
                break
        self.result_queue.put(pred_emotion)
            
        cv2.putText(img, f"EMOTION: {pred_emotion.upper()}", (40, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        
        return frame.from_ndarray(img, format="bgr24")

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]}]}
)

# 5. 화면 레이아웃 분할
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🎥 실시간 웹캠 입력")
    webrtc_ctx = webrtc_streamer(
        key="live-emotion-streamer", 
        video_processor_factory=EmotionProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": {"width": {"ideal": 1280}, "height": {"ideal": 720}}, "audio": False},
        async_processing=True
    )
    
    current_emotion = "neutral"
    if webrtc_ctx and webrtc_ctx.video_processor:
        try:
            current_emotion = webrtc_ctx.video_processor.result_queue.get(timeout=0.1)
        except queue.Empty:
            pass
            
    st.markdown(f"### 📊 현재 감지된 감정: `{current_emotion.upper()}`")

with col2:
    st.subheader("🤖 Gemini 감정 케어 멘토")
    
    if st.button("🔄 현재 내 표정으로 피드백 받기", key="trigger_btn"):
        current_time = time.time()
        # 💡 [우회책 1] 5초 이내에 연속으로 버튼을 누르면 API 호출을 원천 차단
        if current_time - st.session_state.last_call_time < 5.0:
            st.warning("⚠️ 너무 빠르게 버튼을 눌렀습니다! 구글 서버 보호를 위해 5초만 대기 후 눌러주세요.")
        else:
            st.session_state.last_call_time = current_time
            with st.spinner(f"💭 {current_emotion.upper()} 상태를 기반으로 제미나이가 조언을 작성 중입니다..."):
                try:
                    # 💡 [우회책 2] 프롬프트를 고도로 단순화하여 토큰 소모량 최소화 (무료 한도 절약)
                    prompt = f"사용자 감정: {current_emotion}. 이 감정에 맞는 짧은 응원 메시지 2문장으로 작성해줘. 이모지 필수."
                    
                    response = ai_client.models.generate_content(
                        model='gemini-2.0-flash',
                        contents=prompt
                    )
                    
                    st.session_state.chatbot_response = response.text
                    st.session_state.last_processed_emotion = current_emotion.upper()
                    
                except Exception as e:
                    if "429" in str(e):
                        st.session_state.chatbot_response = "⚠️ 구글 API 무료 서버 호출 제한에 걸렸습니다. 완전히 풀릴 때까지 약 10초~30초간 창을 그대로 두고 대기해 주세요."
                    else:
                        st.session_state.chatbot_response = f"⚠️ 호출 오류 발생: {e}"

    # 결과물 출력 영역
    st.info(f"{st.session_state.last_processed_emotion} 감정에 대한 멘토의 편지:")
    st.chat_message("assistant").write(st.session_state.chatbot_response)
