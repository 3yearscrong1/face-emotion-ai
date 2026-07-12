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
from streamlit_webrtc import webrtc_streamer, RTCConfiguration
from google import genai

# 1. 페이지 설정
st.set_page_config(page_title="AI 표정 인식 & 멘토링 챗봇", layout="wide")
st.title("🔮 AI 실시간 표정 인식 & Gemini 피드백 시스템")

# 2. AI 가중치 모델 로드
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
    # PyTorch 2.6+ 보안 경고 및 호환성을 위해 weights_only=False 지정
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

# 3. Gemini API 키 연동
try:
    gemini_key = st.secrets["GEMINI_API_KEY"]
    ai_client = genai.Client(api_key=gemini_key)
except Exception:
    st.error("🔑 Streamlit Secrets 설정을 확인해 주세요.")
    st.stop()

# 세션 상태 변수 초기화
if "chatbot_response" not in st.session_state:
    st.session_state.chatbot_response = "카메라를 켜고 아래 [🔄 피드백 받기] 버튼을 누르면 제미나이 멘토링이 시작됩니다!"
if "current_detected_emotion" not in st.session_state:
    st.session_state.current_detected_emotion = "NEUTRAL"

# 4. 실시간 웹캠 화면 송출을 위한 더미 변환기 (단순 뷰어 역할)
class VideoViewer:
    def transform(self, frame):
        # 복잡한 연산은 여기서 하지 않고 화면만 그대로 송출하여 스레드 부하 제거
        return frame.to_ndarray(format="bgr24")

RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]}]}
)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🎥 실시간 웹캠 입력")
    # webrtc_streamer를 통해 영상 수신 컨텍스트 확보
    webrtc_ctx = webrtc_streamer(
        key="emotion-streamer",
        video_processor_factory=VideoViewer,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": {"width": {"ideal": 1280}, "height": {"ideal": 720}}, "audio": False},
        async_transform=True
    )
    
    st.markdown(f"### 📊 감지된 감정: `{st.session_state.current_detected_emotion}`")

with col2:
    st.subheader("🤖 Gemini 감정 케어 멘토")
    
    if st.button("🔄 현재 내 표정으로 피드백 받기", key="trigger_btn"):
        # 🛠️ [핵심 해결 방법] 버튼을 누른 딱 그 순간의 비디오 락(Lock) 프레임을 직접 스냅샷으로 가져옴
        if webrtc_ctx and webrtc_ctx.video_receiver:
            try:
                # 최신 비디오 프레임 타임스탬프 스냅샷 가져오기
                frame = webrtc_ctx.video_receiver.get_frame()
                if frame is not None:
                    img = frame.to_ndarray(format="bgr24")
                    
                    # AI 감정 모델 연산 수행
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
                    detected_emotion = classes[max_idx]
                    st.session_state.current_detected_emotion = detected_emotion.upper()
                    
                    # 5. Gemini 피드백 생성 요청
                    with st.spinner("💭 제미나이가 표정을 분석하고 답변을 작성 중입니다..."):
                        prompt = f"""
                        사용자의 현재 표정 분석 감정 상태는 [{detected_emotion}] 입니다.
                        이 감정에 맞는 따뜻한 위로, 공감, 혹은 응원의 피드백을 친구처럼 친근한 말투로 딱 2~3문장 이내로 작성해줘.
                        말끝에는 감정에 어울리는 이모지(예시: 😊, 😭, ☕)를 자연스럽게 섞어줘.
                        """
                        response = ai_client.models.generate_content(
                            model='gemini-2.0-flash',
                            contents=prompt
                        )
                        st.session_state.chatbot_response = response.text
                        # 한 번 정상 처리되면 화면 강제 리프레시를 위해 rerun
                        st.rerun()
                else:
                    st.warning("⚠️ 웹캠이 아직 준비되지 않았거나 프레임을 읽지 못했습니다. 잠시 후 다시 눌러주세요.")
            except Exception as e:
                if "429" in str(e):
                    st.session_state.chatbot_response = "⚠️ 구글 무료 서버 제한에 일시적으로 도달했습니다. 10초만 완전히 대기한 후 버튼을 다시 가볍게 눌러주세요!"
                else:
                    st.session_state.chatbot_response = f"연동 중 문제가 발생했습니다: {e}"
        else:
            st.warning("🎥 먼저 왼쪽의 [START] 버튼을 눌러 카메라를 켜주세요!")

    # 결과 출력
    st.info(f"{st.session_state.current_detected_emotion} 감정에 대한 멘토의 편지:")
    st.chat_message("assistant").write(st.session_state.chatbot_response)
