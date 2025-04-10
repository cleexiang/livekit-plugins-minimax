# Copyright 202 LiveKit, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
import json
import aiohttp

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    tts,
    utils,
)
from livekit.agents.types import (
    DEFAULT_API_CONNECT_OPTIONS,
    NOT_GIVEN,
    NotGivenOr,
)
from livekit.agents.utils import is_given

from .log import logger
from .models import (
    TTSModels,
    TTSLanguage,
    TTSDefaultVoiceId,
    TTSSampleRate,
)

@dataclass
class _TTSOptions:
    model: str  # Minimax uses different model names
    voice_id: str  # Replace speaker with voice_id
    sample_rate: int # 
    speed: float  # 
    language: str = "en"
    volume: float = 1.0  # 
    pitch: float = 0.0   # 
    bitrate: int = 128000  # 
    format: str = "mp3"  # 
    channel: int = 1     # 


DEFAULT_API_URL = "https://api.minimaxi.chat/v1/t2a_v2"


NUM_CHANNELS = 1


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        model: TTSModels | str = "speech-01-turbo",
        language: TTSLanguage = "English",
        voice_id: str = TTSDefaultVoiceId,
        sample_rate: TTSSampleRate = 32000,
        speed: float = 1.0,
        volume: float = 1.0,
        pitch: float = 0.0,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        group_id: NotGivenOr[str] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
    ) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(
                streaming=False,
            ),
            sample_rate=sample_rate,
            num_channels=NUM_CHANNELS,
        )
        self._api_key = api_key if is_given(api_key) else os.environ.get("MINIMAX_API_KEY")
        self._group_id = group_id if is_given(group_id) else os.environ.get("MINIMAX_GROUP_ID")
        
        if not self._api_key:
            raise ValueError(
                "Minimax API key is required, either as argument or set MINIMAX_API_KEY environmental variable"
            )
        if not self._group_id:
            raise ValueError(
                "Minimax Group ID is required, either as argument or set MINIMAX_GROUP_ID environmental variable"
            )

        self._opts = _TTSOptions(
            model=model,
            language=language,
            voice_id=voice_id,
            sample_rate=sample_rate,
            speed=speed,
            volume=volume,
            pitch=pitch,
        )
        self._session = http_session

    def _ensure_session(self) -> aiohttp.ClientSession:
        if not self._session:
            self._session = utils.http_context.http_session()

        return self._session

    def synthesize(
        self,
        text: str,
        *,
        conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS,
        segment_id: NotGivenOr[str] = NOT_GIVEN,
    ) -> ChunkedStream:
        assert self._api_key is not None  # validated in constructor
        assert self._group_id is not None  # validated in constructor
        return ChunkedStream(
            tts=self,
            input_text=text,
            conn_options=conn_options,
            opts=self._opts,
            session=self._ensure_session(),
            api_key=self._api_key,
            group_id=self._group_id,
        )

    def update_options(
        self,
        *,
        model: NotGivenOr[str] = NOT_GIVEN,
        voice_id: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        if is_given(model):
            self._opts.model = model
        if is_given(voice_id):
            self._opts.voice_id = voice_id


class ChunkedStream(tts.ChunkedStream):
    """Synthesize using the chunked api endpoint"""

    def __init__(
        self,
        tts: TTS,
        input_text: str,
        opts: _TTSOptions,
        api_key: str,
        group_id: str,
        session: aiohttp.ClientSession,
        conn_options: APIConnectOptions,
        segment_id: NotGivenOr[str] = NOT_GIVEN,
    ) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._opts = opts
        self._session = session
        self._segment_id = segment_id if is_given(segment_id) else utils.shortuuid()
        self._api_key = api_key
        self._group_id = group_id
    async def _run(self) -> None:
        request_id = utils.shortuuid()
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        
        payload = {
            "text": self._input_text,
            "model": self._opts.model,
            "language_boost": self._opts.language,
            "stream": True,
            "voice_setting": {
                "voice_id": self._opts.voice_id,
            },
            "audio_setting": {
                "sample_rate": self._opts.sample_rate,
                "bitrate": self._opts.bitrate,
                "format": self._opts.format,
                "channel": self._opts.channel
            }
        }

        decoder = utils.codecs.AudioStreamDecoder(
            sample_rate=self._opts.sample_rate,
            num_channels=NUM_CHANNELS,
        )

        decode_task: asyncio.Task | None = None
        try:
            url = f"{DEFAULT_API_URL}?GroupId={self._group_id}"
            async with self._session.post(
                url, 
                headers=headers, 
                json=payload,
                read_bufsize=2**16,  # 1MB buffer
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status != 200:
                    content = await response.text()
                    logger.error("Minimax API error: %s", content)
                    return

                content = await response.text()
                for line in content.split('\n'):
                    line = line.strip()
                    if not line or line == 'data: [DONE]':
                        continue
                    if line.startswith('data: '):
                        try:
                            json_str = line[6:]  # Remove 'data: ' prefix
                            response_json = json.loads(json_str)
                            if 'data' in response_json and 'audio' in response_json['data']:
                                audio_data = bytes.fromhex(response_json['data']['audio'])
                                decoder.push(audio_data)
                        except Exception as e:
                            logger.warning(f"Failed to process line: {e}")
                            continue

                decoder.end_input()
                emitter = tts.SynthesizedAudioEmitter(
                    event_ch=self._event_ch,
                    request_id=request_id,
                    segment_id=self._segment_id,
                )
                async for frame in decoder:
                    emitter.push(frame)
                emitter.flush()

        except asyncio.TimeoutError as e:
            raise APITimeoutError() from e
        except aiohttp.ClientResponseError as e:
            raise APIStatusError(
                message=e.message,
                status_code=e.status,
                request_id=request_id,
                body=None,
            ) from e
        except Exception as e:
            raise APIConnectionError() from e
        finally:
            if decode_task:
                await utils.aio.gracefully_cancel(decode_task)
            await decoder.aclose()