import math
import json
import copy # For deepcopying schemas
from typing import Dict, List, Optional, Union, Any

from pydantic import BaseModel, Field # Ensure BaseModel and Field are imported
import tiktoken
import google.generativeai as genai
# GenerativeModel is now accessed via genai.GenerativeModel
from google.generativeai.types import FunctionDeclaration, Tool 

from openai import (
    APIError,
    AsyncAzureOpenAI,
    AsyncOpenAI,
    AuthenticationError,
    OpenAIError,
    RateLimitError,
)
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from app.bedrock import BedrockClient
from app.config import LLMSettings, config
from app.exceptions import TokenLimitExceeded
from app.logger import logger  # Assuming a logger is set up in your app
from app.schema import (
    ROLE_VALUES,
    TOOL_CHOICE_TYPE,
    TOOL_CHOICE_VALUES,
    Message,
    ToolChoice,
    Function as SchemaFunction, # Import Function and alias it
)

# Helper Pydantic models for consistent LLM response structure
class LLMToolCall(BaseModel):
    id: str
    type: str = "function"
    function: SchemaFunction # Use the imported and aliased Function model

class LLMResponseMessage(BaseModel):
    role: str = "assistant"
    content: Optional[str] = None
    tool_calls: Optional[List[LLMToolCall]] = None

    class Config:
        arbitrary_types_allowed = True

# Models like MULTIMODAL_MODELS, REASONING_MODELS are defined here in the actual file.

REASONING_MODELS = ["o1", "o3-mini"]
MULTIMODAL_MODELS = [
    "gpt-4-vision-preview",
    "gpt-4o",
    "gpt-4o-mini",
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
]


class TokenCounter:
    # Token constants
    BASE_MESSAGE_TOKENS = 4
    FORMAT_TOKENS = 2
    LOW_DETAIL_IMAGE_TOKENS = 85
    HIGH_DETAIL_TILE_TOKENS = 170

    # Image processing constants
    MAX_SIZE = 2048
    HIGH_DETAIL_TARGET_SHORT_SIDE = 768
    TILE_SIZE = 512

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def count_text(self, text: str) -> int:
        """Calculate tokens for a text string"""
        return 0 if not text else len(self.tokenizer.encode(text))

    def count_image(self, image_item: dict) -> int:
        """
        Calculate tokens for an image based on detail level and dimensions

        For "low" detail: fixed 85 tokens
        For "high" detail:
        1. Scale to fit in 2048x2048 square
        2. Scale shortest side to 768px
        3. Count 512px tiles (170 tokens each)
        4. Add 85 tokens
        """
        detail = image_item.get("detail", "medium")

        # For low detail, always return fixed token count
        if detail == "low":
            return self.LOW_DETAIL_IMAGE_TOKENS

        # For medium detail (default in OpenAI), use high detail calculation
        # OpenAI doesn't specify a separate calculation for medium

        # For high detail, calculate based on dimensions if available
        if detail == "high" or detail == "medium":
            # If dimensions are provided in the image_item
            if "dimensions" in image_item:
                width, height = image_item["dimensions"]
                return self._calculate_high_detail_tokens(width, height)

        return (
            self._calculate_high_detail_tokens(1024, 1024) if detail == "high" else 1024
        )

    def _calculate_high_detail_tokens(self, width: int, height: int) -> int:
        """Calculate tokens for high detail images based on dimensions"""
        # Step 1: Scale to fit in MAX_SIZE x MAX_SIZE square
        if width > self.MAX_SIZE or height > self.MAX_SIZE:
            scale = self.MAX_SIZE / max(width, height)
            width = int(width * scale)
            height = int(height * scale)

        # Step 2: Scale so shortest side is HIGH_DETAIL_TARGET_SHORT_SIDE
        scale = self.HIGH_DETAIL_TARGET_SHORT_SIDE / min(width, height)
        scaled_width = int(width * scale)
        scaled_height = int(height * scale)

        # Step 3: Count number of 512px tiles
        tiles_x = math.ceil(scaled_width / self.TILE_SIZE)
        tiles_y = math.ceil(scaled_height / self.TILE_SIZE)
        total_tiles = tiles_x * tiles_y

        # Step 4: Calculate final token count
        return (
            total_tiles * self.HIGH_DETAIL_TILE_TOKENS
        ) + self.LOW_DETAIL_IMAGE_TOKENS

    def count_content(self, content: Union[str, List[Union[str, dict]]]) -> int:
        """Calculate tokens for message content"""
        if not content:
            return 0

        if isinstance(content, str):
            return self.count_text(content)

        token_count = 0
        for item in content:
            if isinstance(item, str):
                token_count += self.count_text(item)
            elif isinstance(item, dict):
                if "text" in item:
                    token_count += self.count_text(item["text"])
                elif "image_url" in item:
                    token_count += self.count_image(item)
        return token_count

    def count_tool_calls(self, tool_calls: List[dict]) -> int:
        """Calculate tokens for tool calls"""
        token_count = 0
        for tool_call in tool_calls:
            if "function" in tool_call:
                function = tool_call["function"]
                token_count += self.count_text(function.get("name", ""))
                token_count += self.count_text(function.get("arguments", ""))
        return token_count

    def count_message_tokens(self, messages: List[dict]) -> int:
        """Calculate the total number of tokens in a message list"""
        total_tokens = self.FORMAT_TOKENS  # Base format tokens

        for message in messages:
            tokens = self.BASE_MESSAGE_TOKENS  # Base tokens per message

            # Add role tokens
            tokens += self.count_text(message.get("role", ""))

            # Add content tokens
            if "content" in message:
                tokens += self.count_content(message["content"])

            # Add tool calls tokens
            if "tool_calls" in message:
                tokens += self.count_tool_calls(message["tool_calls"])

            # Add name and tool_call_id tokens
            tokens += self.count_text(message.get("name", ""))
            tokens += self.count_text(message.get("tool_call_id", ""))

            total_tokens += tokens

        return total_tokens


class LLM:
    _instances: Dict[str, "LLM"] = {}

    def __new__(
        cls, config_name: str = "default", llm_config: Optional[LLMSettings] = None
    ):
        if config_name not in cls._instances:
            instance = super().__new__(cls)
            # Initialize the instance immediately after creation and before storing
            instance.__init__(config_name, llm_config)
            cls._instances[config_name] = instance
        return cls._instances[config_name]

    @staticmethod
    def _sanitize_schema_for_gemini(data: Any) -> Any:
        """
        Recursively remove all keys named 'title' or 'default' from a nested dictionary or list of dictionaries.
        This is used to sanitize JSON schemas for APIs like Gemini that don't support these fields in function parameter schemas.
        """
        if isinstance(data, dict):
            new_dict = {}
            for k, v in data.items():
                if k == "title" or k == "default":  # Skip 'title' and 'default' keys
                    continue
                new_dict[k] = LLM._sanitize_schema_for_gemini(v)  # Recurse
            return new_dict
        elif isinstance(data, list):
            return [LLM._sanitize_schema_for_gemini(item) for item in data]
        else:
            return data

    def __init__(
        self, config_name: str = "default", llm_config: Optional[LLMSettings] = None):
            # logger.critical(f"LLM.__INIT__ ENTERED. config_name='{config_name}'")

            if not hasattr(self, "client"):  # Only initialize if not already initialized
                # Load configuration
                loaded_llm_config_source = llm_config
                if not llm_config:
                    # Assuming 'config' is the global AppConfig instance from app.config
                    # If config is not always imported at module level, ensure it's available here
                    from app.config import config as global_app_config
                    loaded_llm_config_source = global_app_config.llm

                effective_llm_settings = loaded_llm_config_source.get(config_name, loaded_llm_config_source.get("default"))

                if effective_llm_settings is None:
                    logger.error("LLM.__INIT__: FAILED to load effective_llm_settings. Aborting.")
                    raise ValueError("Could not load LLM settings.")

                self.model = effective_llm_settings.model
                self.max_tokens = effective_llm_settings.max_tokens
                self.temperature = effective_llm_settings.temperature
                self.api_type = effective_llm_settings.api_type
                self.api_key = effective_llm_settings.api_key
                self.api_version = effective_llm_settings.api_version
                self.base_url = effective_llm_settings.base_url

                # logger.critical(f"LLM.__INIT__: EFFECTIVE API_TYPE = '{self.api_type}' from config.")
                logger.info(f"LLM.__init__: Initializing LLM client with api_type='{self.api_type}', model='{self.model}'") # Simplified log

                # Add token counting related attributes
                self.total_input_tokens = 0
                self.total_completion_tokens = 0
                self.max_input_tokens = (
                    effective_llm_settings.max_input_tokens
                    if hasattr(effective_llm_settings, "max_input_tokens")
                    else None
                )

                # Initialize tokenizer
                try:
                    self.tokenizer = tiktoken.encoding_for_model(self.model)
                except KeyError:
                    # If the model is not in tiktoken's presets, use cl100k_base as default
                    self.tokenizer = tiktoken.get_encoding("cl100k_base")

                if self.api_type == "azure":
                    logger.info("LLM.__init__: Configuring for AZURE.")
                    self.client = AsyncAzureOpenAI(
                        base_url=self.base_url,
                        api_key=self.api_key,
                        api_version=self.api_version,
                    )
                elif self.api_type == "aws":
                    logger.info("LLM.__init__: Configuring for AWS BEDROCK.")
                    # Assuming BedrockClient is defined elsewhere, e.g., app.bedrock
                    # from app.bedrock import BedrockClient # If not already imported
                    # self.client = BedrockClient() # Replace with actual Bedrock client init
                    logger.warning("LLM.__init__: AWS Bedrock client initialization is a placeholder.")
                    # self.client = BedrockClient() # Example
                    pass # Placeholder until BedrockClient is fully integrated
                elif self.api_type == "google":
                    logger.info("LLM.__init__: Configuring for GOOGLE/GEMINI.")
                    try:
                        genai.configure(api_key=self.api_key)
                        self.client = genai.GenerativeModel(model_name=self.model)
                        logger.info(f"LLM.__init__: Successfully configured Gemini client. Type: {type(self.client)}")
                    except Exception as e:
                        logger.error(f"LLM.__init__: FAILED to configure Gemini client: {e}", exc_info=True)
                        raise
                elif self.api_type == "openai" or not self.api_type: # Default to OpenAI if api_type is 'openai' or empty/None
                    logger.info(f"LLM.__init__: Configuring for OPENAI (api_type: '{self.api_type}').")
                    self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)
                else:
                    logger.error(f"LLM.__init__: Unsupported api_type: '{self.api_type}'")
                    raise ValueError(f"Unsupported api_type: {self.api_type}")
            
                # logger.info(f"LLM.__init__: Final self.client type: {type(self.client)}")
                self.token_counter = TokenCounter(self.tokenizer)
            else:
                logger.info("LLM.__INIT__: Client already initialized.")

    def count_tokens(self, text: str) -> int:
        """Calculate the number of tokens in a text"""
        if not text:
            return 0
        return len(self.tokenizer.encode(text))

    def count_message_tokens(self, messages: List[dict]) -> int:
        return self.token_counter.count_message_tokens(messages)

    def update_token_count(self, input_tokens: int, completion_tokens: int = 0) -> None:
        """Update token counts"""
        # Only track tokens if max_input_tokens is set
        self.total_input_tokens += input_tokens
        self.total_completion_tokens += completion_tokens
        logger.info(
            f"Token usage: Input={input_tokens}, Completion={completion_tokens}, "
            f"Cumulative Input={self.total_input_tokens}, Cumulative Completion={self.total_completion_tokens}, "
            f"Total={input_tokens + completion_tokens}, Cumulative Total={self.total_input_tokens + self.total_completion_tokens}"
        )

    def check_token_limit(self, input_tokens: int) -> bool:
        """Check if token limits are exceeded"""
        if self.max_input_tokens is not None:
            return (self.total_input_tokens + input_tokens) <= self.max_input_tokens
        # If max_input_tokens is not set, always return True
        return True

    def get_limit_error_message(self, input_tokens: int) -> str:
        """Generate error message for token limit exceeded"""
        if (
            self.max_input_tokens is not None
            and (self.total_input_tokens + input_tokens) > self.max_input_tokens
        ):
            return f"Request may exceed input token limit (Current: {self.total_input_tokens}, Needed: {input_tokens}, Max: {self.max_input_tokens})"

        return "Token limit exceeded"

    @staticmethod
    def format_messages(
        messages: List[Union[dict, Message]], supports_images: bool = False
    ) -> List[dict]:
        """
        Format messages for LLM by converting them to OpenAI message format.

        Args:
            messages: List of messages that can be either dict or Message objects
            supports_images: Flag indicating if the target model supports image inputs

        Returns:
            List[dict]: List of formatted messages in OpenAI format

        Raises:
            ValueError: If messages are invalid or missing required fields
            TypeError: If unsupported message types are provided

        Examples:
            >>> msgs = [
            ...     Message.system_message("You are a helpful assistant"),
            ...     {"role": "user", "content": "Hello"},
            ...     Message.user_message("How are you?")
            ... ]
            >>> formatted = LLM.format_messages(msgs)
        """
        formatted_messages = []

        for message in messages:
            # Convert Message objects to dictionaries
            if isinstance(message, Message):
                message = message.to_dict()

            if isinstance(message, dict):
                # If message is a dict, ensure it has required fields
                if "role" not in message:
                    raise ValueError("Message dict must contain 'role' field")

                # Process base64 images if present and model supports images
                if supports_images and message.get("base64_image"):
                    # Initialize or convert content to appropriate format
                    if not message.get("content"):
                        message["content"] = []
                    elif isinstance(message["content"], str):
                        message["content"] = [
                            {"type": "text", "text": message["content"]}
                        ]
                    elif isinstance(message["content"], list):
                        # Convert string items to proper text objects
                        message["content"] = [
                            (
                                {"type": "text", "text": item}
                                if isinstance(item, str)
                                else item
                            )
                            for item in message["content"]
                        ]

                    # Add the image to content
                    message["content"].append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{message['base64_image']}"
                            },
                        }
                    )

                    # Remove the base64_image field
                    del message["base64_image"]
                # If model doesn't support images but message has base64_image, handle gracefully
                elif not supports_images and message.get("base64_image"):
                    # Just remove the base64_image field and keep the text content
                    del message["base64_image"]

                if "content" in message or "tool_calls" in message:
                    formatted_messages.append(message)
                # else: do not include the message
            else:
                raise TypeError(f"Unsupported message type: {type(message)}")

        # Validate all messages have required fields
        for msg in formatted_messages:
            if msg["role"] not in ROLE_VALUES:
                raise ValueError(f"Invalid role: {msg['role']}")

        return formatted_messages

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_exception_type(
            (OpenAIError, Exception, ValueError)
        ),  # Don't retry TokenLimitExceeded
    )
    async def ask(
        self,
        messages: List[Union[dict, Message]],
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        stream: bool = True,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Send a prompt to the LLM and get the response.

        Args:
            messages: List of conversation messages
            system_msgs: Optional system messages to prepend
            stream (bool): Whether to stream the response
            temperature (float): Sampling temperature for the response

        Returns:
            str: The generated response

        Raises:
            TokenLimitExceeded: If token limits are exceeded
            ValueError: If messages are invalid or response is empty
            OpenAIError: If API call fails after retries
            Exception: For unexpected errors
        """
        try:
            # Check if the model supports images
            supports_images = self.model in MULTIMODAL_MODELS

            # Format system and user messages with image support check
            if system_msgs:
                system_msgs = self.format_messages(system_msgs, supports_images)
                messages = system_msgs + self.format_messages(messages, supports_images)
            else:
                messages = self.format_messages(messages, supports_images)

            # Calculate input token count
            input_tokens = self.count_message_tokens(messages)

            # Check if token limits are exceeded
            if not self.check_token_limit(input_tokens):
                error_message = self.get_limit_error_message(input_tokens)
                # Raise a special exception that won't be retried
                raise TokenLimitExceeded(error_message)

            params = {
                "model": self.model,
                "messages": messages,
            }

            if self.model in REASONING_MODELS:
                params["max_completion_tokens"] = self.max_tokens
            else:
                params["max_tokens"] = self.max_tokens
                params["temperature"] = (
                    temperature if temperature is not None else self.temperature
                )

            if not stream:
                # Non-streaming request
                response = await self.client.chat.completions.create(
                    **params, stream=False
                )

                if not response.choices or not response.choices[0].message.content:
                    raise ValueError("Empty or invalid response from LLM")

                # Update token counts
                self.update_token_count(
                    response.usage.prompt_tokens, response.usage.completion_tokens
                )

                return response.choices[0].message.content

            # Streaming request, For streaming, update estimated token count before making the request
            self.update_token_count(input_tokens)

            response = await self.client.chat.completions.create(**params, stream=True)

            collected_messages = []
            completion_text = ""
            async for chunk in response:
                chunk_message = chunk.choices[0].delta.content or ""
                collected_messages.append(chunk_message)
                completion_text += chunk_message
                print(chunk_message, end="", flush=True)

            print()  # Newline after streaming
            full_response = "".join(collected_messages).strip()
            if not full_response:
                raise ValueError("Empty response from streaming LLM")

            # estimate completion tokens for streaming response
            completion_tokens = self.count_tokens(completion_text)
            logger.info(
                f"Estimated completion tokens for streaming response: {completion_tokens}"
            )
            self.total_completion_tokens += completion_tokens

            return full_response

        except TokenLimitExceeded:
            # Re-raise token limit errors without logging
            raise
        except ValueError:
            logger.exception(f"Validation error")
            raise
        except OpenAIError as oe:
            logger.exception(f"OpenAI API error")
            if isinstance(oe, AuthenticationError):
                logger.error("Authentication failed. Check API key.")
            elif isinstance(oe, RateLimitError):
                logger.error("Rate limit exceeded. Consider increasing retry attempts.")
            elif isinstance(oe, APIError):
                logger.error(f"API error: {oe}")
            raise
        except Exception:
            logger.exception(f"Unexpected error in ask")
            raise

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_exception_type(
            (OpenAIError, Exception, ValueError)
        ),  # Don't retry TokenLimitExceeded
    )
    async def ask_with_images(
        self,
        messages: List[Union[dict, Message]],
        images: List[Union[str, dict]],
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        stream: bool = False,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Send a prompt with images to the LLM and get the response.

        Args:
            messages: List of conversation messages
            images: List of image URLs or image data dictionaries
            system_msgs: Optional system messages to prepend
            stream (bool): Whether to stream the response
            temperature (float): Sampling temperature for the response

        Returns:
            str: The generated response

        Raises:
            TokenLimitExceeded: If token limits are exceeded
            ValueError: If messages are invalid or response is empty
            OpenAIError: If API call fails after retries
            Exception: For unexpected errors
        """
        try:
            # For ask_with_images, we always set supports_images to True because
            # this method should only be called with models that support images
            if self.model not in MULTIMODAL_MODELS:
                raise ValueError(
                    f"Model {self.model} does not support images. Use a model from {MULTIMODAL_MODELS}"
                )

            # Format messages with image support
            formatted_messages = self.format_messages(messages, supports_images=True)

            # Ensure the last message is from the user to attach images
            if not formatted_messages or formatted_messages[-1]["role"] != "user":
                raise ValueError(
                    "The last message must be from the user to attach images"
                )

            # Process the last user message to include images
            last_message = formatted_messages[-1]

            # Convert content to multimodal format if needed
            content = last_message["content"]
            multimodal_content = (
                [{"type": "text", "text": content}]
                if isinstance(content, str)
                else content
                if isinstance(content, list)
                else []
            )

            # Add images to content
            for image in images:
                if isinstance(image, str):
                    multimodal_content.append(
                        {"type": "image_url", "image_url": {"url": image}}
                    )
                elif isinstance(image, dict) and "url" in image:
                    multimodal_content.append({"type": "image_url", "image_url": image})
                elif isinstance(image, dict) and "image_url" in image:
                    multimodal_content.append(image)
                else:
                    raise ValueError(f"Unsupported image format: {image}")

            # Update the message with multimodal content
            last_message["content"] = multimodal_content

            # Add system messages if provided
            if system_msgs:
                all_messages = (
                    self.format_messages(system_msgs, supports_images=True)
                    + formatted_messages
                )
            else:
                all_messages = formatted_messages

            # Calculate tokens and check limits
            input_tokens = self.count_message_tokens(all_messages)
            if not self.check_token_limit(input_tokens):
                raise TokenLimitExceeded(self.get_limit_error_message(input_tokens))

            # Set up API parameters
            params = {
                "model": self.model,
                "messages": all_messages,
                "stream": stream,
            }

            # Add model-specific parameters
            if self.model in REASONING_MODELS:
                params["max_completion_tokens"] = self.max_tokens
            else:
                params["max_tokens"] = self.max_tokens
                params["temperature"] = (
                    temperature if temperature is not None else self.temperature
                )

            # Handle non-streaming request
            if not stream:
                response = await self.client.chat.completions.create(**params)

                if not response.choices or not response.choices[0].message.content:
                    raise ValueError("Empty or invalid response from LLM")

                self.update_token_count(response.usage.prompt_tokens)
                return response.choices[0].message.content

            # Handle streaming request
            self.update_token_count(input_tokens)
            response = await self.client.chat.completions.create(**params)

            collected_messages = []
            async for chunk in response:
                chunk_message = chunk.choices[0].delta.content or ""
                collected_messages.append(chunk_message)
                print(chunk_message, end="", flush=True)

            print()  # Newline after streaming
            full_response = "".join(collected_messages).strip()

            if not full_response:
                raise ValueError("Empty response from streaming LLM")

            return full_response

        except TokenLimitExceeded:
            raise
        except ValueError as ve:
            logger.error(f"Validation error in ask_with_images: {ve}")
            raise
        except OpenAIError as oe:
            logger.error(f"OpenAI API error: {oe}")
            if isinstance(oe, AuthenticationError):
                logger.error("Authentication failed. Check API key.")
            elif isinstance(oe, RateLimitError):
                logger.error("Rate limit exceeded. Consider increasing retry attempts.")
            elif isinstance(oe, APIError):
                logger.error(f"API error: {oe}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error in ask_with_images: {e}")
            raise

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(6),
        retry=retry_if_exception_type(
            (OpenAIError, Exception, ValueError)
        ),  # Don't retry TokenLimitExceeded
    )
    async def ask_tool(
        self,
        messages: List[Union[dict, Message]],
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        timeout: int = 300,
        tools: Optional[List[dict]] = None,
        tool_choice: TOOL_CHOICE_TYPE = ToolChoice.AUTO,  # type: ignore
        temperature: Optional[float] = None,
        **kwargs,
) -> Any:  # Return type will vary based on API (e.g. ChatCompletionMessage or Gemini response dict)
        """
        Ask LLM using functions/tools and return the response.

        Args:
            messages: List of conversation messages
            system_msgs: Optional system messages to prepend
            timeout: Request timeout in seconds (may not be applicable to all APIs)
            tools: List of tools to use (expected format may vary by API)
            tool_choice: Tool choice strategy (interpretation may vary by API)
            temperature: Sampling temperature for the response
            **kwargs: Additional completion arguments

        Returns:
            Model's response, structure depends on the API used.

        Raises:
            TokenLimitExceeded: If token limits are exceeded (if checked before API-specific logic)
            ValueError: If tools, tool_choice, or messages are invalid (if checked before API-specific logic)
            Exception: For API-specific errors or unexpected issues.
        """
        # logger.critical(f"ASK_TOOL ENTERED. Effective self.api_type='{getattr(self, 'api_type', 'NOT_SET')}'. Client is type: {type(getattr(self, 'client', None))}")

        # --- COMMON VALIDATIONS AND PREP (IF ANY) ---
        # Example: tool_choice validation (if its meaning is somewhat generic)
        if tool_choice not in TOOL_CHOICE_VALUES and not isinstance(tool_choice, dict): # Adjusted for potential dict tool_choice for specific functions
             logger.warning(f"Potentially invalid tool_choice format: {tool_choice}")
             # raise ValueError(f\"Invalid tool_choice: {tool_choice}\") # Or handle per API

        # Token limit checks and message formatting are often API-specific,
        # so they will largely be handled within the respective API blocks.

        if self.api_type == "google":
            logger.info("ASK_TOOL: Executing GOOGLE specific path.")
            try:
                # logger.info("ASK_TOOL (Google Path): Preparing to call Gemini.")

                # 1. Format messages for Gemini
                gemini_messages_for_api = []
                # Combine system messages and regular messages
                all_input_messages = (system_msgs or []) + messages
                if not all_input_messages:
                    logger.error("ASK_TOOL (Google Path): No messages provided to send to Gemini.")
                    raise ValueError("Messages list is empty for Gemini call")

                for msg_item in all_input_messages:
                    # Gemini expects 'user' or 'model' roles. Adapt your Message object's role.
                    # Assuming msg_item has 'role' and 'content' attributes.
                    # This is a basic mapping; adjust if your Message structure is different.
                    role = "user" if msg_item.role in ["user", "system"] else "model"
                    if not hasattr(msg_item, 'content') or not msg_item.content:
                        logger.warning(f"ASK_TOOL (Google Path): Skipping message with no content: {msg_item}")
                        continue
                    gemini_messages_for_api.append({'role': role, 'parts': [{'text': msg_item.content}]})

                if not gemini_messages_for_api: # Check again after filtering
                    logger.error("ASK_TOOL (Google Path): No valid messages to send after formatting for Gemini.")
                    raise ValueError("No valid messages to send to Gemini after formatting.")

                # 2. Format tools for Gemini
                gemini_tool_declarations = []
                if tools:
                    for mcp_tool_param in tools:
                        fn_data = mcp_tool_param.get("function")
                        if fn_data and fn_data.get("name") and fn_data.get("parameters") is not None: # Ensure parameters key exists
                            original_schema = fn_data["parameters"]
                            # logger.info(f"ASK_TOOL (Google Path): Processing tool '{fn_data['name']}'. Original parameters schema: {json.dumps(original_schema, indent=2)}")
                            
                            # Sanitize the schema by removing "title" and "default" fields
                            sanitized_schema = LLM._sanitize_schema_for_gemini(original_schema)
                            # logger.info(f"ASK_TOOL (Google Path): Tool '{fn_data['name']}'. Sanitized parameters schema for Gemini: {json.dumps(sanitized_schema, indent=2)}")

                            # Sanitize tool name for Gemini
                            original_tool_name = fn_data["name"]
                            simple_tool_name = original_tool_name # Default to original

                            # Define mappings from observed suffixes to simple names
                            # These simple names should match actual tool capabilities and Gemini requirements.
                            TOOL_NAME_SUFFIX_MAP = {
                                "_ba": "bash",
                                "_br": "browser_use", 
                                "_st": "str_replace_editor",
                                "_te": "terminate"
                            }
                            
                            # Attempt to map using suffix if the name seems mangled
                            # Check for a length that suggests it's a mangled name before trying suffix.
                            # A simple heuristic: if it contains '/' or is very long.
                            # For now, we'll rely on the suffix for known mangled forms.
                            mapped = False
                            for suffix, mapped_name in TOOL_NAME_SUFFIX_MAP.items():
                                if original_tool_name.endswith(suffix):
                                    simple_tool_name = mapped_name
                                    mapped = True
                                    break
                            
                            if mapped:
                                # logger.info(f"ASK_TOOL (Google Path): Mapped tool name '{original_tool_name}' to '{simple_tool_name}' for Gemini.")
                                pass # Name was mapped, proceed
                            elif simple_tool_name != original_tool_name: # If it was changed by some other logic not shown
                                # logger.info(f"ASK_TOOL (Google Path): Using tool name '{simple_tool_name}' (from original '{original_tool_name}') for Gemini.")
                                pass
                            else: # No mapping applied, using original name (might be an issue if still mangled or invalid)
                                logger.warning(f"ASK_TOOL (Google Path): No specific mapping for tool name '{original_tool_name}'. Using as is. Ensure it is valid for Gemini.")

                            # Final validation check (basic) - Gemini API will perform strict validation
                            if not (simple_tool_name.replace('_', '').replace('-', '').replace('.', '').isalnum() and \
                                    len(simple_tool_name) <= 63 and \
                                    (simple_tool_name[0].isalpha() or simple_tool_name[0] == '_')):
                                logger.error(f"ASK_TOOL (Google Path): Sanitized tool name '{simple_tool_name}' (from original '{original_tool_name}') is STILL LIKELY INVALID for Gemini due to length or characters. Length: {len(simple_tool_name)}")
                                # Optionally, skip this tool or raise an error to prevent API call with known bad name
                                # continue # Skips adding this tool declaration
                            
                            gemini_tool_declarations.append(
                                FunctionDeclaration(
                                    name=simple_tool_name, # Use the sanitized/mapped name
                                    description=fn_data.get("description", "No description provided."),
                                    parameters=sanitized_schema, # Use the sanitized schema
                                )
                            )
                        else:
                            logger.warning(f"ASK_TOOL (Google Path): Skipping malformed tool for Gemini: Name={fn_data.get('name') if fn_data else 'N/A'}, Params_Present={fn_data.get('parameters') is not None if fn_data else False}")
                
                gemini_tools_list_for_api = [Tool(function_declarations=gemini_tool_declarations)] if gemini_tool_declarations else None
                # logger.info(f"ASK_TOOL (Google Path): Tools for Gemini: {gemini_tools_list_for_api}")

                # 3. Generation Config
                generation_config_params = {}
                if temperature is not None: generation_config_params["temperature"] = temperature
                # Gemini uses 'max_output_tokens'. Ensure 'self.max_tokens' is appropriate.
                if self.max_tokens: generation_config_params["max_output_tokens"] = self.max_tokens
                final_generation_config = genai.types.GenerationConfig(**generation_config_params) if generation_config_params else None

                logger.info(f"ASK_TOOL (Google Path): Calling generate_content_async with model {self.client.model_name if hasattr(self.client, 'model_name') else 'N/A'}")
                
                if not isinstance(self.client, genai.GenerativeModel): # Check type
                    logger.error(f"ASK_TOOL (Google Path): self.client is NOT a GenerativeModel. It is {type(self.client)}")
                    raise TypeError("Incorrect client type for Gemini API call.")

                response = await self.client.generate_content_async(
                    contents=gemini_messages_for_api,
                    tools=gemini_tools_list_for_api,
                    generation_config=final_generation_config
                )
                logger.info("ASK_TOOL (Google Path): Received response from Gemini.")

                # 4. Process Gemini Response (Simplified - adapt to your agent's needs)
                # This should map to a structure similar to ChatCompletionMessage if possible
                # For now, returning a dictionary.
                if response.candidates and response.candidates[0].content and response.candidates[0].content.parts:
                    api_response_part = response.candidates[0].content.parts[0]
                    if hasattr(api_response_part, 'function_call') and api_response_part.function_call:
                        fc = api_response_part.function_call
                        logger.info(f"ASK_TOOL (Google Path): Gemini returned function call: {fc.name}")
                        # Map to your agent's expected tool_calls structure
                        tool_calls_list = [
                            LLMToolCall(
                                id=fc.name, # Or generate a unique ID if needed by your framework
                                type="function",
                                function=SchemaFunction(name=fc.name, arguments=json.dumps(dict(fc.args)))
                            )
                        ]
                        return LLMResponseMessage(tool_calls=tool_calls_list, content=None)
                    elif hasattr(api_response_part, 'text'):
                        logger.info("ASK_TOOL (Google Path): Gemini returned text response.")
                        return LLMResponseMessage(content=api_response_part.text, tool_calls=None)

                logger.warning("ASK_TOOL (Google Path): Gemini response was empty or not in expected format.")
                # Return a consistent error structure if needed by the agent, wrapped appropriately
                return LLMResponseMessage(content="Error: Empty or unparseable response from Gemini.", tool_calls=None)

            except Exception as e: # Catch specific google.api_core.exceptions if possible
                logger.error(f"ASK_TOOL (Google Path): Error during Gemini API call: {e}", exc_info=True)
                raise # Re-raise to be handled by the agent or a global handler

        else: # Non-Google path (OpenAI, Azure, potentially unhandled)
            logger.info(f"ASK_TOOL: Executing NON-GOOGLE (Else) path for api_type \'{self.api_type}\'.")
            try:
                # This block should contain the logic for OpenAI, Azure, etc.
                # It's based on the original structure that was leading to errors when misconfigured for Gemini.

                # Validate tool_choice (if not done globally or if OpenAI specific values are different)
                # if tool_choice not in TOOL_CHOICE_VALUES: # Assuming TOOL_CHOICE_VALUES is for OpenAI
                #     raise ValueError(f\"Invalid tool_choice for OpenAI-like API: {tool_choice}\")

                supports_images = self.model in MULTIMODAL_MODELS

                # Format messages for OpenAI-like APIs
                formatted_messages = []
                if system_msgs:
                    formatted_system_msgs = self.format_messages(system_msgs, supports_images)
                    formatted_messages.extend(formatted_system_msgs)
                formatted_messages.extend(self.format_messages(messages, supports_images))

                if not formatted_messages:
                    raise ValueError("No messages to send after formatting for OpenAI-like API.")

                # Calculate input token count for OpenAI-like APIs
                # input_tokens = self.count_message_tokens(formatted_messages)
                # tools_tokens_count = 0
                # if tools:
                #     for tool_item in tools: tools_tokens_count += self.count_tokens(str(tool_item))
                # input_tokens += tools_tokens_count
                # if not self.check_token_limit(input_tokens):
                #     raise TokenLimitExceeded(self.get_limit_error_message(input_tokens))

                # Validate tools structure for OpenAI-like APIs
                if tools:
                    for tool_item in tools:
                        if not isinstance(tool_item, dict) or "type" not in tool_item or tool_item["type"] != "function" or "function" not in tool_item:
                            raise ValueError("Each tool for OpenAI-like API must be a dict with type \'function\' and a \'function\' object.")

                params = {
                    "model": self.model,
                    "messages": formatted_messages,
                    "tools": tools, # Assumes \'tools\' is already in OpenAI format
                    "tool_choice": tool_choice, # Assumes \'tool_choice\' is compatible
                    "timeout": timeout,
                    **kwargs,
                }

                if self.model in REASONING_MODELS:
                    params["max_completion_tokens"] = self.max_tokens
                else:
                    params["max_tokens"] = self.max_tokens
                current_temp = temperature if temperature is not None else self.temperature
                if current_temp is not None: # Ensure temperature is only added if set
                    params["temperature"] = current_temp

                params["stream"] = False # Non-streaming for tool calls

                logger.info(f"ASK_TOOL (Non-Google Path): Calling OpenAI compatible API with param keys: {list(params.keys())}")

                if not isinstance(self.client, AsyncOpenAI) and not isinstance(self.client, AsyncAzureOpenAI) : # Check client type
                     logger.error(f"ASK_TOOL (Non-Google Path): Expected AsyncOpenAI/AsyncAzureOpenAI client but got {type(self.client)} for api_type \'{self.api_type}\'")
                     raise TypeError(f"Misconfigured client for api_type \'{self.api_type}\'. Client is {type(self.client)}")

                response: ChatCompletion = await self.client.chat.completions.create(**params)

                if not response.choices or not response.choices[0].message:
                    logger.error(f"ASK_TOOL (Non-Google Path): Invalid or empty response from LLM: {response}")
                    return None # Or raise an error

                # Update token counts (if successful)
                if response.usage:
                    self.update_token_count(
                        response.usage.prompt_tokens, response.usage.completion_tokens
                    )

                return response.choices[0].message # This is a ChatCompletionMessage

            except TokenLimitExceeded: # Catch specific exceptions first
                # logger.error("ASK_TOOL (Non-Google Path): Token limit exceeded.") # Already logged by check_token_limit usually
                raise
            except ValueError as ve:
                logger.error(f"ASK_TOOL (Non-Google Path): Validation error: {ve}", exc_info=True)
                raise
            except OpenAIError as oe: # This is where the original error was logged (lines 756, 762)
                logger.error(f"ASK_TOOL (Non-Google Path): OpenAI API error: {oe}", exc_info=True)
                # Specific OpenAI error type logging can be re-added here if desired
                # if isinstance(oe, AuthenticationError): logger.error("...")
                # elif isinstance(oe, RateLimitError): logger.error("...")
                # elif isinstance(oe, APIError): logger.error("...")
                raise
            except Exception as e:
                logger.error(f"ASK_TOOL (Non-Google Path): Unexpected error: {e}", exc_info=True)
                raise
