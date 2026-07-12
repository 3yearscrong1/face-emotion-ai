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
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
from google import genai
import queue

# 1. 페이지 레이아웃 세팅
st.set_page_config(page_title="AI 표정 인식 & 멘토링 챗봇", layout="wide")
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

# 3. 구글 API 키 가져오기
try:
    gemini_key = st.secrets["GEMINI_API_KEY"]
    ai_client = genai.Client(api_key=gemini_key)
except Exception:
    st.error("🔑 Streamlit Secrets 설정을 확인해 주세요.")
    st.stop()

# 챗봇 답변 및 감정 스냅샷용 세션 변수
if "chatbot_response" not in st.session_state:
    st.session_state.chatbot_response = "★ 시스템이 성공적으로 리셋되었습니다! 버튼을 눌러 피드백을 요청하세요. ★"
if "last_detected_emotion" not in st.session_state:
    st.session_state.last_detected_emotion = "neutral"

# 4. 웹캠 비디오 프레임 처리 클래스
class EmotionTransformer(VideoTransformerBase):
    def __init__(self):
        self.result_queue = queue.Queue()

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # 명암비 equalization으로 이목구비 및 스마일 라인 인식률 대폭 강화
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
        
        # 큐 비우고 최신 감정 데이터 매칭
        while not self.result_queue.empty():
            try:
                self.result_queue.get_nowait()
            except queue.Empty:
                break
        self.result_queue.put(pred_emotion)
            
        # 화면에 예측된 감정 실시간 텍스트 출력
        cv2.putText(img, f"EMOTION: {pred_emotion.upper()}", (40, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        
        return img

# STUN 서버 설정
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]}]}
)

# 5. 화면 레이아웃 분할
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🎥 실시간 웹캠 입력")
    webrtc_ctx = webrtc_streamer(
        key="emotion-streamer", 
        video_transformer_factory=EmotionTransformer,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": {"width": {"ideal": 1280}, "height": {"ideal": 720}}, "audio": False},
        async_transform=True
    )
    
    # 🛠️ [속성 에러 완벽 해결] video_transformer와 video_processor 버전을 둘 다 안전하게 예외처리 탐색
    processor = None
    if webrtc_ctx:
        if hasattr(webrtc_ctx, "video_transformer") and webrtc_ctx.video_transformer:
            processor = webrtc_ctx.video_transformer
        elif hasattr(webrtc_ctx, "video_processor") and webrtc_ctx.video_processor:
            processor = webrtc_ctx.video_processor

    # 안전하게 큐에서 최신 감정 상태 획득
    if processor and hasattr(processor, "result_queue"):
        try:
            st.session_state.last_detected_emotion = processor.result_queue.get(timeout=0.1)
        except queue.Empty:
            pass
            
    current_emo_display = st.session_state.last_detected_emotion.upper()
    st.markdown(f"### 📊 현재 감지된 감정: `{current_emo_display}`")

with col2:
    st.subheader("🤖 Gemini 감정 케어 멘토")
    
    if st.button("🔄 현재 내 표정으로 피드백 받기", key="trigger_btn"):
        with st.spinner("💭 제미나이가 표정을 분석하고 멘토링 답변을 작성 중입니다..."):
            try:
                # 동기화된 스레드 안전 변수 확보
                target_emotion = st.session_state.last_detected_emotion
                
                prompt = f"""
                사용자의 현재 표정 분석 감정 상태는 [{target_emotion}] 입니다.
                이 감정에 맞는 따뜻한 위로, 공감, 혹은 응원의 피드백을 친구처럼 친근한 말투로 딱 2~3문장 이내로 작성해줘.
                말끝에는 감정에 어울리는 이모지(예시: 😊, 😭, ☕)를 자연스럽게 섞어줘.
                """
                response = ai_client.models.generate_content(
                    model='gemini-2.0-flash',
                    contents=prompt
                )
                st.session_state.chatbot_response = response.text
            except Exception as e:
                if "429" in str(e):
                    st.session_state.chatbot_response = "⚠️ 구글 무료 서버 제한에 일시적으로 도달했습니다. 5~10초만 완전히 대기한 후 버튼을 다시 가볍게 눌러주세요!"
                else:
                    st.session_state.chatbot_response = f"연동 오류 발생: {e}"

    # 결과물 출력 영역
    st.info(f"{st.session_state.last_detected_emotion.upper()} 감정에 대한 멘토의 편지:")
    st.chat_message("assistant").write(st.session_state.chatbot_response)
