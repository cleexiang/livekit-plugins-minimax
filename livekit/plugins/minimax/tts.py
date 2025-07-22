# Copyright 2025 LiveKit, Inc.
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
import base64
import ssl

from livekit.agents import (
    APIConnectionError,
    APIConnectOptions,
    APIStatusError,
    APITimeoutError,
    tokenize,
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
from .models import TTSModels
import weakref

BUFFERED_WORDS_COUNT = 3
DEFAULT_MINIMAX_WEBSOCKET_URL = "wss://api.minimaxi.chat/ws/v1/t2a_v2"
DEFAULT_API_URL = "https://api.minimax.chat/v1/t2a_v2"
NUM_CHANNELS = 1
MODULE = "speech-02-hd"
EMOTION = "happy"

@dataclass
class _TTSOptions:
    model: str  # Minimax uses different model names
    voice_id: str  # Replace speaker with voice_id
    sample_rate: int # 
    speed: float  # 
    tokenizer: tokenize.SentenceTokenizer
    language: str = "en"
    volume: float = 1.0  # 
    pitch: float = 0.0   # 
    bitrate: int = 128000  # 
    format: str = "mp3"  # 
    channel: int = 1   # 


class TTS(tts.TTS):
    def __init__(
        self,
        *,
        model: str = "speech-01-turbo",
        language: str = "en",
        voice_id: str = "Santa_Claus",
        sample_rate: int = 32000,
        speed: float = 1.0,
        volume: float = 1.0,
        pitch: float = 0.0,
        api_url: str = DEFAULT_API_URL,
        api_key: NotGivenOr[str] = NOT_GIVEN,
        group_id: NotGivenOr[str] = NOT_GIVEN,
        http_session: aiohttp.ClientSession | None = None,
        tokenizer: tokenize.SentenceTokenizer | None = None,
        use_streaming: bool = False,
    ) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(
                streaming=use_streaming,
            ),
            sample_rate=sample_rate,
            num_channels=NUM_CHANNELS,
        )
        self._api_url = api_url
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
        if tokenizer is None:
            tokenizer = tokenize.basic.SentenceTokenizer(min_sentence_len=BUFFERED_WORDS_COUNT)

        self._opts = _TTSOptions(
            model=model,
            language=language,
            voice_id=voice_id,
            sample_rate=sample_rate,
            speed=speed,
            tokenizer=tokenizer,
            volume=volume,
            pitch=pitch,
        )
        self._session = http_session
        self._streams = weakref.WeakSet[SynthesizeStream]()
        self._pool = utils.ConnectionPool[aiohttp.ClientWebSocketResponse](
            connect_cb=self._connect_ws,
            close_cb=self._close_ws,
        )

    async def _connect_ws(self) -> aiohttp.ClientWebSocketResponse:
        session = self._ensure_session()
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        logger.info(f"Connecting to Minimax TTS WebSocket: {DEFAULT_MINIMAX_WEBSOCKET_URL}")
        return await asyncio.wait_for(
            session.ws_connect(
                DEFAULT_MINIMAX_WEBSOCKET_URL,
                headers={"Authorization": f"Bearer {self._api_key}"},
                ssl=ssl_context
            ),
            
            self._conn_options.timeout,
        )

    async def _close_ws(self, ws: aiohttp.ClientWebSocketResponse):
        await ws.send_str(json.dumps({"event": "task_finish"}))
        await ws.close()
        logger.info("Minimax TTS WebSocket closed")
        
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
            api_url=self._api_url,
            api_key=self._api_key,
            group_id=self._group_id,
        )
        
    def stream(self, *, conn_options: APIConnectOptions | None = None) -> SynthesizeStream:
        assert self._api_key is not None  # validated in constructor
        assert self._group_id is not None  # validated in constructor
        stream = SynthesizeStream(
            tts=self,
            pool=self._pool,
            opts=self._opts,
            api_key=self._api_key,
        )
        self._streams.add(stream)
        return stream


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
        api_url: str,
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
        self._api_url = api_url
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
            "language": self._opts.language,
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
            url = f"{self._api_url}?GroupId={self._group_id}"
            logger.info(f"Starting Minimax TTS request for text of length {len(self._input_text)}")
            start_time = asyncio.get_event_loop().time()
            first_response_time = None
            
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

                # 直接读取整个响应内容
                content = await response.text()
                for line in content.split('\n'):
                    line = line.strip()
                    if not line or line == 'data: [DONE]':
                        continue
                    if line.startswith('data: '):
                        try:
                            if first_response_time is None:
                                first_response_time = asyncio.get_event_loop().time()
                                first_response_elapsed = first_response_time - start_time
                                logger.info(f"Received first Minimax TTS response in {first_response_elapsed:.2f} seconds")
                            
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
                
            end_time = asyncio.get_event_loop().time()
            elapsed_time = end_time - start_time
            logger.info(f"Completed Minimax TTS request in {elapsed_time:.2f} seconds")

        except asyncio.TimeoutError as e:
            logger.error("Minimax TTS request timed out")
            raise APITimeoutError() from e
        except aiohttp.ClientResponseError as e:
            logger.error(f"Minimax TTS request failed with status {e.status}: {e.message}")
            raise APIStatusError(
                message=e.message,
                status_code=e.status,
                request_id=request_id,
                body=None,
            ) from e
        except Exception as e:
            logger.error(f"Minimax TTS request failed with error: {str(e)}")
            raise APIConnectionError() from e
        finally:
            if decode_task:
                await utils.aio.gracefully_cancel(decode_task)
            await decoder.aclose()
            
class SynthesizeStream(tts.SynthesizeStream):
    def __init__(
        self,
        *,
        tts: TTS,
        opts: _TTSOptions,
        pool: utils.ConnectionPool[aiohttp.ClientWebSocketResponse],
        api_key: str,
    ):
        super().__init__(tts=tts)
        self._opts = opts
        self._pool = pool
        self._api_key = api_key
        self._group_id = tts._group_id
        
    async def _run(self) -> None:
        request_id = utils.shortuuid()
        self._segments_ch = utils.aio.Chan[tokenize.SentenceStream]()

        @utils.log_exceptions(logger=logger)
        async def _tokenize_input():
            """tokenize text from the input_ch to words"""
            input_stream = None
            async for input in self._input_ch:
                if isinstance(input, str):
                    if input_stream is None:
                        # new segment (after flush for e.g)
                        input_stream = self._opts.tokenizer.stream()
                        self._segments_ch.send_nowait(input_stream)
                    input_stream.push_text(input)
                elif isinstance(input, self._FlushSentinel):
                    if input_stream is not None:
                        input_stream.end_input()
                    input_stream = None
            if input_stream is not None:
                input_stream.end_input()
            self._segments_ch.close()

        @utils.log_exceptions(logger=logger)
        async def _process_segments():
            async for input_stream in self._segments_ch:
                await self._run_ws(input_stream)

        tasks = [
            asyncio.create_task(_tokenize_input()),
            asyncio.create_task(_process_segments()),
        ]
        try:
            await asyncio.gather(*tasks)
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
            await utils.aio.gracefully_cancel(*tasks)

    async def _run_ws(
        self,
        input_stream: tokenize.SentenceStream,
    ) -> None:
        async with self._pool.connection() as ws:
            response = json.loads(await ws.receive_str())
            logger.info(f"Minimax TTS response: {response}")
            if response.get("event") != "connected_success":
                raise APIConnectionError(f"Failed to connect to Minimax TTS: {response}")

            segment_id = utils.shortuuid()
            decoder = utils.codecs.AudioStreamDecoder(
                sample_rate=self._opts.sample_rate,
                num_channels=NUM_CHANNELS,
            )
            index_lock = asyncio.Lock()
            current_index = 0
            pending_requests = set()

            # Add a coordination event
            task_started = asyncio.Event()
            
            @utils.log_exceptions(logger=logger)
            async def _send_task(ws: aiohttp.ClientWebSocketResponse):
                # First send task_start message
                start_msg = {
                    "event": "task_start",
                    "model": self._opts.model,
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
                # Send task_start and wait for confirmation
                logger.info("Sending task_start to Minimax TTS")
                await ws.send_str(json.dumps(start_msg))

                msg = await ws.receive()
                if msg.type != aiohttp.WSMsgType.TEXT:
                    raise APIConnectionError(f"Unexpected response type from Minimax TTS: {msg.type}")
                
                response = json.loads(msg.data)
                logger.info(f"Minimax TTS response: {response}")
                if response.get("event") != "task_started":
                    raise APIConnectionError(f"Failed to start Minimax TTS task: {response}")
                
                # Signal that task has started successfully
                task_started.set()
                
                nonlocal current_index
                index = 0
                # Now send text data using task_continue event
                async for data in input_stream:
                    continue_msg = {
                        "event": "task_continue",
                        "text": data.token
                    }
                    async with index_lock:
                        pending_requests.add(index)
                    index += 1
                    current_index = index
                    await ws.send_str(json.dumps(continue_msg))

            @utils.log_exceptions(logger=logger)
            async def _recv_task(ws: aiohttp.ClientWebSocketResponse):
                # Wait until task_start confirmation before processing audio
                await task_started.wait()
                
                while True:
                    msg = await ws.receive()
                    if msg.type in (
                        aiohttp.WSMsgType.CLOSED,
                        aiohttp.WSMsgType.CLOSE,
                        aiohttp.WSMsgType.CLOSING,
                    ):
                        raise APIStatusError(
                            "Minimax connection closed unexpectedly",
                            request_id=str(current_index),
                        )

                    if msg.type != aiohttp.WSMsgType.TEXT:
                        logger.warning("Unexpected Minimax message type %s", msg.type)
                        continue

                    response = json.loads(msg.data)
                    
                    if "data" in response and "audio" in response["data"]:
                        audio = response["data"]["audio"]
                        audio_bytes = bytes.fromhex(audio)
                        decoder.push(audio_bytes)
                        
                    if response.get("is_final"):
                        decoder.end_input()
                        break
                        # async with index_lock:
                        #     index = current_index
                        #     pending_requests.remove(index)
                        #     if not pending_requests:
                        #         decoder.end_input()
                        #         break  # we are not going to receive any more audio
                    else:
                        logger.debug("Received Minimax message: %s", response.get("trace_id"))

            @utils.log_exceptions(logger=logger)
            async def _emit_task():
                emitter = tts.SynthesizedAudioEmitter(
                    event_ch=self._event_ch,
                    request_id=str(current_index),
                    segment_id=segment_id,
                )
                async for frame in decoder:
                    emitter.push(frame)
                emitter.flush()

            tasks = [
                asyncio.create_task(_send_task(ws)),
                asyncio.create_task(_recv_task(ws)),
                asyncio.create_task(_emit_task()),
            ]

            try:
                await asyncio.gather(*tasks)
            except asyncio.TimeoutError as e:
                raise APITimeoutError() from e
            except aiohttp.ClientResponseError as e:
                raise APIStatusError(
                    message=e.message,
                    status_code=e.status,
                    request_id=str(current_index),
                    body=None,
                ) from e
            except Exception as e:
                raise APIConnectionError() from e
            finally:
                await utils.aio.gracefully_cancel(*tasks)