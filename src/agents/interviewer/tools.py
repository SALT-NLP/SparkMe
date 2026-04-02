import asyncio
import time
from typing import Dict, Type, Optional, Any, Callable
from langchain_core.callbacks.manager import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, SkipValidation

from src.utils.speech.text_to_speech import TextToSpeechBase, create_tts_engine
from src.utils.speech.audio_player import create_audio_player, AudioPlayerBase
from src.utils.constants.colors import RED, RESET, GREEN


class ResponseToUserInput(BaseModel):
    subtopic_id: str = Field(description="The chosen subtopic ID from the suggested subtopics.")
    response: str = Field(description="The response to the user.")

class RespondToUser(BaseTool):
    """Tool for responding to the user."""
    name: str = "respond_to_user"
    description: str = "A tool for responding to the user."
    args_schema: Type[BaseModel] = ResponseToUserInput
    
    tts_config: Dict = Field(default_factory=dict)
    base_path: str = Field(...)
    on_response: SkipValidation[Callable[[str], None]] = Field(
        description="Callback function to be called when responding to user"
    )
    on_turn_complete: SkipValidation[Callable[[], None]] = Field(
        description="Callback function to be called when turn is complete"
    )
    tts_engine: Optional[Any] = Field(default=None, exclude=True)
    audio_player: Optional[Any] = Field(default=None, exclude=True)

    def __init__(self, **data):
        super().__init__(**data)
        if self.tts_config.get("enabled", False):
            self.tts_engine: TextToSpeechBase = create_tts_engine(
                provider=self.tts_config.get("provider", "openai"),
                voice=self.tts_config.get("voice", "alloy")
            )
            self.audio_player: AudioPlayerBase = create_audio_player()

    async def _tts_and_play(self, text: str, output_path: Optional[str] = None) -> None:
        # Wait for TTS to complete
        audio_path = self.tts_engine.text_to_speech(text=text, output_path=output_path)

        # Print and play audio in background
        asyncio.create_task(self._print_and_play(audio_path))

    async def _print_and_play(self, audio_path: str) -> None:
        print(f"{GREEN}Audio saved to: {audio_path}{RESET}")
        await asyncio.to_thread(self.audio_player.play, audio_path)

    async def _run(
        self,
        subtopic_id: str,
        response: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> Any:
        quantified_response = await self.on_response(response, subtopic_id)

        if self.tts_engine:
            try:
                await self._tts_and_play(
                    quantified_response,
                    f"{self.base_path}/audio_outputs/response_{int(time.time())}.mp3"
                )
            except Exception as e:
                print(f"{RED}Failed to generate/play speech: {e}{RESET}")
        
        self.on_turn_complete()
            
        return "Response sent to the user."