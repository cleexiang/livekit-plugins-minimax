from typing import Literal

TTSEncoding = Literal[
    "pcm_s16le",
    # Not yet supported
    # "pcm_f32le",
    # "pcm_mulaw",
    # "pcm_alaw",
]

TTSModels = Literal["speech-01-turbo", "speech-01-hd"]
TTSLanguages = Literal["en", "es", "fr", "de", "pt", "zh", "ja"]
TTSDefaultVoiceId = "Friendly_Person"
TTSVoiceEmotion = Literal[
    "happy",
    "sad",
    "angry",
    "fearful"
]