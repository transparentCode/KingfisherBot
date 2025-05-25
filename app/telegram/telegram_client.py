import logging
import os
import asyncio
from concurrent.futures.thread import ThreadPoolExecutor

import jinja2
from enum import Enum
from typing import Optional, List, Union
from dataclasses import dataclass
import threading
from pathlib import Path

from dotenv import load_dotenv
from aiogram import Bot, Dispatcher
from aiogram.enums import ParseMode

load_dotenv()

telegram_chat_id = os.getenv("TELEGRAM_CHAT_ID")
telegram_allowed_users = os.getenv("TELEGRAM_ALLOWED_USERS")
telegram_bot_token = os.getenv("TELEGRAM_BOT_TOKEN")
telegram_templates_dir = os.getenv("TELEGRAM_TEMPLATES_DIR")


class MessageFormat(Enum):
    PLAIN = "plain"
    HTML = "HTML"
    MARKDOWN = "MARKDOWN"
    MARKDOWN_V2 = "MARKDOWN_V2"


@dataclass
class TelegramConfig:
    def __init__(self):
        self.chat_ids = [telegram_chat_id]
        self.allowed_users = [telegram_allowed_users]
        self.token = telegram_bot_token
        self.log_level = 'INFO'
        self.template_dir = os.path.join(os.path.dirname(__file__), telegram_templates_dir)
        self.default_format = MessageFormat.MARKDOWN


class MessageTemplate:
    def __init__(self, template_dir: str):
        self.env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(template_dir),
            autoescape=True,
            trim_blocks=True,
            lstrip_blocks=True
        )

    def render(self, template_name: str, **kwargs) -> str:
        try:
            template = self.env.get_template(template_name)
            return template.render(**kwargs)
        except jinja2.TemplateNotFound:
            raise ValueError(f"Template {template_name} not found")
        except Exception as e:
            raise Exception(f"Template rendering error: {str(e)}")


class MessageFormatter:
    @staticmethod
    def format_message(text: str, format_type: MessageFormat) -> tuple[str, Optional[str]]:
        if format_type == MessageFormat.PLAIN:
            return text, None
        elif format_type == MessageFormat.HTML:
            return text, ParseMode.HTML
        elif format_type == MessageFormat.MARKDOWN:
            return text, ParseMode.MARKDOWN
        elif format_type == MessageFormat.MARKDOWN_V2:
            special_chars = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+',
                             '-', '=', '|', '{', '}', '.', '!']
            for char in special_chars:
                text = text.replace(char, f'\\{char}')
            return text, ParseMode.MARKDOWN_V2


class TelegramClient:
    _instance = None
    _lock = asyncio.Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TelegramClient, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config: Optional[TelegramConfig] = None):
        if getattr(self, '_initialized', False):
            return

        self._config = TelegramConfig() if not config else config
        self.logger = logging.getLogger('app')

        # Initialize bot with aiogram 3.x
        self.bot = Bot(token=self._config.token)
        self.dp = Dispatcher()  # No bot parameter in v3.x

        self._initialized = True
        self.logger = logging.getLogger(__name__)
        self.template_engine = MessageTemplate(self._config.template_dir)

        # Create a thread executor for running async tasks
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._loop = asyncio.new_event_loop()

        # Start the event loop in a separate thread
        self._thread = threading.Thread(target=self._start_loop, daemon=True)
        self._thread.start()

    def _start_loop(self):
        """Run event loop in the thread"""
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def _run_coroutine(self, coro):
        """Helper method to run a coroutine and get its result"""
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return future.result(timeout=10)  # Add timeout to prevent hanging
        except Exception as e:
            self.logger.error(f"Error running coroutine: {str(e)}")
            return False

    async def ensure_bot_connection(self):
        """Ensure bot connection is alive and reinitialize if needed"""
        try:
            # Test connection
            await self.bot.get_me()
        except Exception as e:
            self.logger.warning(f"Bot connection test failed: {str(e)}. Reinitializing...")
            await self.restart_bot()

    async def restart_bot(self):
        """Properly restart the bot connection"""
        try:
            # Close existing connections if any
            await self.bot.session.close()  # This might be different in v3
            # Reinitialize bot
            self.bot = Bot(token=self._config.token)
            self.dp = Dispatcher()  # No bot parameter
        except Exception as e:
            self.logger.error(f"Failed to restart bot: {str(e)}")
            raise

    def send_message(self, message: str, chat_id: Optional[str] = None,
                     format_type: Optional[MessageFormat] = None, **kwargs) -> bool:
        """Send message using aiogram"""
        format_type = format_type or self._config.default_format
        text, parse_mode = MessageFormatter.format_message(message, format_type)

        async def _send():
            try:
                if chat_id:
                    await self.bot.send_message(
                        chat_id=chat_id,
                        text=text,
                        parse_mode=parse_mode
                    )
                else:
                    for cid in self._config.chat_ids:
                        await self.bot.send_message(
                            chat_id=cid,
                            text=text,
                            parse_mode=parse_mode
                        )
                return True
            except Exception as e:
                self.logger.error(f"Failed to send message: {str(e)}")
                try:
                    await self.restart_bot()
                except Exception as restart_error:
                    self.logger.error(f"Failed to restart bot after send failure: {str(restart_error)}")
                return False

        return self._run_coroutine(_send())

    def send_template(self, template_name: str, chat_id: Optional[str] = None,
                      format_type: Optional[MessageFormat] = None, **kwargs) -> bool:
        """Send message from template"""
        try:
            message = self.template_engine.render(template_name, **kwargs)
            return self.send_message(
                message=message,
                chat_id=chat_id,
                format_type=format_type
            )
        except Exception as e:
            self.logger.error(f"Failed to send template message: {str(e)}")
            return False

    def send_photo(self, photo_path: str, caption: Optional[str] = None,
                   chat_id: Optional[str] = None) -> bool:
        """Send photo using aiogram"""

        async def _send_photo():
            try:
                # Instead of opening the file yourself, use FSInputFile from aiogram
                from aiogram.types import FSInputFile
                photo = FSInputFile(photo_path)

                if chat_id:
                    await self.bot.send_photo(
                        chat_id=chat_id,
                        photo=photo,
                        caption=caption
                    )
                else:
                    for cid in self._config.chat_ids:
                        await self.bot.send_photo(
                            chat_id=cid,
                            photo=photo,
                            caption=caption
                        )
                return True
            except Exception as e:
                self.logger.error(f"Failed to send photo: {str(e)}")
                return False

        return self._run_coroutine(_send_photo())

    def send_document(self, file_path: str, caption: Optional[str] = None,
                      chat_id: Optional[str] = None) -> bool:
        """Send document using aiogram"""

        async def _send_document():
            try:
                with open(file_path, 'rb') as document:
                    if chat_id:
                        await self.bot.send_document(
                            chat_id=chat_id,
                            document=document,
                            caption=caption
                        )
                    else:
                        for cid in self._config.chat_ids:
                            await self.bot.send_document(
                                chat_id=cid,
                                document=document,
                                caption=caption
                            )
                return True
            except Exception as e:
                self.logger.error(f"Failed to send document: {str(e)}")
                return False

        return self._run_coroutine(_send_document())

    def send_chart_alert(self, template_name: str, chat_id: Optional[str] = None,
                         chart_path: Optional[str] = None, **kwargs) -> bool:
        """Send a chart alert with template message"""
        try:
            # Indicate in template context that we have a chart
            if chart_path:
                kwargs['chart_image'] = True

            # Render the template
            message = self.template_engine.render(template_name, **kwargs)

            if chart_path and os.path.exists(chart_path):
                # If we have a chart image, send it with caption
                return self.send_photo(
                    photo_path=chart_path,
                    caption=message,
                    chat_id=chat_id
                )
            else:
                # Otherwise just send the message
                return self.send_message(
                    message=message,
                    chat_id=chat_id,
                    format_type=MessageFormat.HTML
                )
        except Exception as e:
            self.logger.error(f"Failed to send chart alert: {str(e)}")
            return False

    def __del__(self):
        """Clean up resources when the instance is destroyed"""
        if hasattr(self, '_loop') and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if hasattr(self, '_executor'):
            self._executor.shutdown(wait=False)