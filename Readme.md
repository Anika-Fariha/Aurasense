# 🌌 AuraSense Multimodal AI Agent for Emotion Detection Powered by LLM


AuraSense detects human emotions from video, audio, and text and generates context-aware insights using an LLM.

## Features
- **Vision**: Recognizes facial expressions
- **Audio**: Detects tone, stress, and emotion
- **Text**: Understands sentiment and context
- **Fusion**: Combines all modalities for accurate emotion detection
- **AI Models**: DINOv2, Wav2Vec2, DistilBERT, and Whisper integration

#### **MELD Dataset (Primary Evaluation)**

#### **RAVDESS Dataset (Transfer Learning Study)**

### **Ablation Studies**


### **Multimodal Fusion**

    Trimodal Fusion Architecture
    class TrimodalFusionModel(nn.Module):
    def init(self):
    super().init()
    self.vision_compress = nn.Linear(768, 256) # Noise reduction
    self.cross_attention = CrossAttentionLayer(768)
    self.fusion_layers = nn.Sequential(
    nn.Linear(768 + 768 + 256, 512),
    nn.Dropout(0.3),
    nn.ReLU(),
    nn.Linear(512, 128),
    nn.Dropout(0.2),
    nn.Linear(128, 5) # 5 emotion classes
    )

    def forward(self, text_feat, audio_feat, vision_feat):
        # Vision compression (breakthrough technique)
        vision_compressed = self.vision_compress(vision_feat)
        
        # Cross-modal attention
        text_attended = self.cross_attention(text_feat, audio_feat, vision_compressed)
        audio_attended = self.cross_attention(audio_feat, text_feat, vision_compressed)
        
        # Feature concatenation and prediction
        fused = torch.cat([text_attended, audio_attended, vision_compressed], dim=1)
        emotion_logits = self.fusion_layers(fused)
        
        return F.softmax(emotion_logits, dim=1)


