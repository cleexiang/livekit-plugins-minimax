from typing import Literal

TTSModels = Literal[
    "speech-02-hd-preview",
    "speech-02-turbo-preview",
    "speech-01-hd",
    "speech-01-turbo",
]

TTSLanguage = Literal["Chinese", "Chinese,Yue", "English", "Arabic", "Russian", "Spanish", "French", "Portuguese", "German", "Turkish", "Dutch", "Ukrainian", "Vietnamese", "Indonesian", "Japanese", "Italian", "Korean", "Thai", "Polish", "Romanian", "Greek", "Czech", "Finnish", "Hindi", "auto"]

TTSSampleRate = Literal[8000, 16000, 22050, 24000, 32000, 44100]

TTSEncoding = Literal["mp3", "pcm", "flac"]

TTSDefaultVoiceId = "Friendly_Person"
