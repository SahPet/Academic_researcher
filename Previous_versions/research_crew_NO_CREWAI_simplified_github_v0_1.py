import os
import json
import logging
import asyncio
import openai
from dotenv import load_dotenv
from datetime import datetime
from serper_tool import SerperSearchTool
import requests
from bs4 import BeautifulSoup
from io import BytesIO
from pypdf import PdfReader
import re
import random
import aiohttp
from typing import Tuple, Dict, Set, Any, List, Optional
from urllib.parse import urlparse
from html import unescape
from pydantic import BaseModel
import semanticscholar as sch  # Add this import line
import habanero  # Add this import for Crossref API

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

import html2text
import httpx
from scrapingbee import ScrapingBeeClient
import base64
import json
import time
import chardet
import fitz

from google import genai
from google.genai.types import GenerateContentConfig
import json
import time
from openai import OpenAI
import httpx
import os
from enum import Enum
import traceback

# Model selection configuration
class ModelChoice(Enum):
    O3MINI = "o3mini"
    GEMINI = "gemini"
    DEEPSEEK = "deepseek"

# Set your preferred model here
PREFERRED_MODEL = ModelChoice.O3MINI  # Change this to .GEMINI or .DEEPSEEK as needed

# Load environment variables
load_dotenv()

# Initialize API keys from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_FLASH_THINKING_ID = os.getenv("GEMINI_FLASH_THINKING_ID", "gemini-2.0-flash-thinking-exp-01-21")
FIREWORKS_API_KEY = os.getenv("FIREWORKS_API_KEY")
SCRAPINGBEE_API_KEY = os.getenv("SCRAPINGBEE_API_KEY")
PARSEHUB_API_KEY = os.getenv("PARSEHUB_API_KEY")
PARSEHUB_PROJECT_TOKEN = os.getenv("PARSEHUB_PROJECT_TOKEN")


# Validate required API keys
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables")


# Initialize clients
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY  # Set for OpenAI client
gemini_client = genai.Client(api_key=GEMINI_API_KEY)
GEMINI_FLASH_THINKING_ID="gemini-2.0-flash-thinking-exp-01-21"


# Initialize root logger at module level
root_logger = logging.getLogger()
root_logger.setLevel(logging.DEBUG)



# Move these variables to the top but don't set LOG_FILE yet
BASE_OUTPUT_FOLDER = os.path.join("C:\\", "research_outputs")
OUTPUT_FOLDER = None
LOG_FOLDER = None
CONTENT_FOLDER = None
SEARCH_FOLDER = None

# Search configuration
MAX_SEARCH_ROUNDS = 7
MAX_REFERENCES = 12

from logging.handlers import RotatingFileHandler

def setup_logging():
    """Set up logging configuration using a rotating file handler."""
    global LOG_FILE
    LOG_FILE = os.path.join(LOG_FOLDER, f"research_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    root_logger.handlers.clear()
    
    # Set the root logger level to DEBUG to capture everything
    root_logger.setLevel(logging.DEBUG)
    
    # Use RotatingFileHandler (max 10 MB per file, with up to 5 backups)
    file_handler = RotatingFileHandler(LOG_FILE, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8')
    file_formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)  # Ensure file handler captures all levels
    root_logger.addHandler(file_handler)
    
    # Console handler with different formatting and level
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.INFO)  # Console shows INFO and above
    root_logger.addHandler(console_handler)
    
    # Log initial setup message
    root_logger.info(f"Logging initialized: {LOG_FILE}")
    root_logger.debug("Debug logging enabled")

def setup_folders():
    """Set up and create necessary folders"""
    global OUTPUT_FOLDER, LOG_FOLDER, CONTENT_FOLDER, SEARCH_FOLDER
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    OUTPUT_FOLDER = os.path.join(BASE_OUTPUT_FOLDER, f"research_session_{timestamp}")
    LOG_FOLDER = os.path.join(OUTPUT_FOLDER, "logs")
    CONTENT_FOLDER = os.path.join(OUTPUT_FOLDER, "content")
    SEARCH_FOLDER = os.path.join(OUTPUT_FOLDER, "search_results")
    
    # Log folder creation
    print(f"Creating folders for session {timestamp}")
    
    for folder in [OUTPUT_FOLDER, LOG_FOLDER, CONTENT_FOLDER, SEARCH_FOLDER]:
        os.makedirs(folder, exist_ok=True)
        
    # Initialize session-specific files with correct structure
    session_files = {
        "session_search_results.json": {
            "metadata": {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "session_id": os.path.basename(OUTPUT_FOLDER),
                "total_results": 0
            },
            "url_map": {}
        },
        "session_picked_references.json": [],
        "session_citations_check.json": []
    }
    
    for filename, initial_data in session_files.items():
        filepath = os.path.join(SEARCH_FOLDER, filename)
        if not os.path.exists(filepath):
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(initial_data, f, indent=2)
    
    # Set up logging after folders are created
    setup_logging()
    
    # Log successful setup
    root_logger.info(f"Session folders created at: {OUTPUT_FOLDER}")
    root_logger.debug(f"Log folder: {LOG_FOLDER}")
    root_logger.debug(f"Content folder: {CONTENT_FOLDER}")
    root_logger.debug(f"Search results folder: {SEARCH_FOLDER}")



def sanitize_json_content(content: str, strict: bool = True) -> str:
    """
    Clean and optionally validate a JSON-like string.
    If the string starts with { or [, assume it is meant to be JSON.
    """
    if not isinstance(content, str):
        return str(content)
    content = content.strip()
    # If it doesn't look like JSON, return it as is
    if not (content.startswith('{') or content.startswith('[')):
        return content

    # Remove markdown code blocks and headers
    content = re.sub(r'```.*?```', '', content, flags=re.DOTALL)
    content = re.sub(r'#.*?\n', '', content)
    content = content.strip()

    if strict:
        try:
            # Try parsing and then re-stringifying
            parsed = json.loads(content)
            return json.dumps(parsed, ensure_ascii=False)
        except json.JSONDecodeError as e:
            root_logger.debug(f"sanitize_json_content: JSON parsing failed: {e}")
    return content


def clean_special_chars(text: str) -> str:
    """Clean special characters from text, including Unicode box-drawing characters."""
    # Replace Unicode box-drawing characters and other special formatting
    replacements = {
        '\u2500': '-',  # ─
        '\u2502': '|',  # │
        '\u2503': '|',  # ┃
        '\u2504': '-',  # ┄
        '\u2505': '-',  # ┅
        '\u2506': '|',  # ┆
        '\u2507': '|',  # ┇
        '\u2508': '-',  # ┈
        '\u2509': '-',  # ┉
        '\u250A': '|',  # ┊
        '\u250B': '|',  # ┋
        '\u250C': '+',  # ┌
        '\u250D': '+',  # ┍
        '\u250E': '+',  # ┎
        '\u250F': '+',  # ┏
        '\u2510': '+',  # ┐
        '\u2511': '+',  # ┑
        '\u2512': '+',  # ┒
        '\u2513': '+',  # ┓
        '\u2514': '+',  # └
        '\u2515': '+',  # ┕
        '\u2516': '+',  # ┖
        '\u2517': '+',  # ┗
        '\u2518': '+',  # ┘
        '\u2519': '+',  # ┙
        '\u251A': '+',  # ┚
        '\u251B': '+',  # ┛
        '\u251C': '+',  # ├
        '\u251D': '+',  # ┝
        '\u251E': '+',  # ┞
        '\u251F': '+',  # ┟
        '\u2520': '+',  # ┠
        '\u2521': '+',  # ┡
        '\u2522': '+',  # ┢
        '\u2523': '+',  # ┣
        '\u2524': '+',  # ┤
        '\u2525': '+',  # ┥
        '\u2526': '+',  # ┦
        '\u2527': '+',  # ┧
        '\u2528': '+',  # ┨
        '\u2529': '+',  # ┩
        '\u252A': '+',  # ┪
        '\u252B': '+',  # ┫
        '\u252C': '+',  # ┬
        '\u252D': '+',  # ┭
        '\u252E': '+',  # ┮
        '\u252F': '+',  # ┯
        '\u2530': '+',  # ┰
        '\u2531': '+',  # ┱
        '\u2532': '+',  # ┲
        '\u2533': '+',  # ┳
        '\u2534': '+',  # ┴
        '\u2535': '+',  # ┵
        '\u2536': '+',  # ┶
        '\u2537': '+',  # ┷
        '\u2538': '+',  # ┸
        '\u2539': '+',  # ┹
        '\u253A': '+',  # ┺
        '\u253B': '+',  # ┻
        '\u253C': '+',  # ┼
        '\u253D': '+',  # ┽
        '\u253E': '+',  # ┾
        '\u253F': '+',  # ┿
        '\u2540': '+',  # ╀
        '\u2541': '+',  # ╁
        '\u2542': '+',  # ╂
        '\u2543': '+',  # ╃
        '\u2544': '+',  # ╄
        '\u2545': '+',  # ╅
        '\u2546': '+',  # ╆
        '\u2547': '+',  # ╇
        '\u2548': '+',  # ╈
        '\u2549': '+',  # ╉
        '\u254A': '+',  # ╊
        '\u254B': '+',  # ╋
        '\u2003': ' ',  # EM SPACE
        '\u25ab': '-',  # WHITE SMALL SQUARE
        '\u25aa': '-',  # BLACK SMALL SQUARE
        '\u2022': '-',  # BULLET
    }
    
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    
    # Remove any remaining unicode characters
    text = text.encode('ascii', 'ignore').decode('ascii')
    
    return text


def validate_response(content: str, min_length: int = 50, is_search_query: bool = False) -> bool:
    """
    Return True if content is nonempty, long enough, and does not indicate an error.
    For search queries, uses different validation rules.
    """
    if not content:
        return False
    
    # Special handling for search queries
    if is_search_query:
        if len(content.strip()) < 1:  # Search queries just need to be non-empty
            return False
        if "[error]" in content.lower():
            return False
        return True
    
    # Standard validation for non-search-query content
    if len(content.strip()) < min_length:
        return False
    if "[error]" in content.lower():
        return False
    try:
        # If it appears to be JSON, try parsing it after sanitization.
        if content.strip().startswith(('{', '[')):
            json.loads(sanitize_json_content(content, strict=True))
    except Exception as e:
        root_logger.debug(f"validate_response: JSON check failed: {e}")
    return True

def flatten_messages(messages: List[Dict[str, str]]) -> str:
    """
    Convert a list of message dictionaries into a single formatted string.
    """
    flattened = []
    for msg in messages:
        role = msg["role"].upper()
        content = msg["content"]
        if role == "SYSTEM":
            flattened.append(f"Instructions: {content}")
        elif role == "USER":
            flattened.append(f"User: {content}")
        elif role == "ASSISTANT":
            flattened.append(f"Assistant: {content}")
    return "\n".join(flattened)



def update_search_results(new_results: list) -> Dict:
    """
    Update session search results additively.
    Preserves existing results while adding new ones.
    Returns: Dict of all accumulated results
    """
    results_file = os.path.join(OUTPUT_FOLDER, "search_results", "session_search_results.json")
    
    # Load existing results
    existing_results = {}
    if os.path.exists(results_file):
        with open(results_file, 'r', encoding='utf-8') as f:
            try:
                file_content = json.load(f)
                # Handle both possible structures (direct or nested in url_map)
                existing_results = file_content.get('url_map', file_content)
            except json.JSONDecodeError:
                root_logger.error("Error reading existing results file")
                existing_results = {}
    
    # Update with new results
    for search_item in new_results:
        results = search_item.get('results', {})
        if isinstance(results, str):
            try:
                results = json.loads(results)
            except json.JSONDecodeError:
                continue
                
        # Handle both possible result structures
        results_list = []
        if isinstance(results, dict):
            results_list = results.get('results', [])
        elif isinstance(results, list):
            results_list = results
                
        for entry in results_list:
            url = entry.get('url')
            if url:
                if url in existing_results:
                    # Append new query to existing entry's history
                    if 'queries' not in existing_results[url]:
                        existing_results[url]['queries'] = []
                    if search_item.get('query') not in existing_results[url]['queries']:
                        existing_results[url]['queries'].append(search_item.get('query'))
                    # Update metadata if new info available
                    if 'metadata' not in existing_results[url]:
                        existing_results[url]['metadata'] = {}
                    existing_results[url]['metadata'].update(entry)
                else:
                    # Create new entry
                    existing_results[url] = {
                        'metadata': entry,
                        'queries': [search_item.get('query')],
                        'first_found': datetime.now().isoformat(),
                        'session_id': os.path.basename(OUTPUT_FOLDER)
                    }
    
    # Save accumulated results with metadata
    output_data = {
        "metadata": {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "session_id": os.path.basename(OUTPUT_FOLDER),
            "total_results": len(existing_results),
            "last_query": new_results[-1].get('query') if new_results else None
        },
        "url_map": existing_results
    }
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)
    
    return existing_results

def log_session_data(data: dict, filename: str):
    """
    Log session-specific data additively.
    """
    filepath = os.path.join(SEARCH_FOLDER, filename)  # Use SEARCH_FOLDER directly
    
    # Load existing data
    existing_data = []
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            existing_data = json.load(f)
    
    # Add new data with timestamp and session ID
    data['timestamp'] = datetime.now().isoformat()
    data['session_id'] = os.path.basename(OUTPUT_FOLDER)
    existing_data.append(data)
    
    # Save updated data
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(existing_data, f, indent=2)

# Replace the custom print function with a proper logging wrapper
def log_print(*args: Any, **kwargs: Any) -> None:
    """
    Wrapper function that both logs and prints messages.
    """
    output = " ".join(map(str, args))
    root_logger.info(output)

# Replace all instances of print with log_print
print = log_print

# Initialize search tool
search_tool = SerperSearchTool()

# Add these constants near other configurations
MAX_CONCURRENT_REQUESTS = 5
REQUEST_TIMEOUT = 90  # seconds - increased from 30





def call_o3mini(messages, model="o3-mini", temperature=0.1, fast_mode=False, max_retries=1, base_delay=2, timeout=90):
    """
    Enhanced o3-mini caller with retry logic, timeout, and response sanitization.
    
    Args:
        messages: List of message dictionaries
        model: Model identifier (default: "o3-mini")
        temperature: Temperature for generation (ignored as not supported by o3-mini)
        fast_mode: If True, uses faster configuration (ignored for now as o3-mini is already fast)
        max_retries: Maximum number of retry attempts (default: 1)
        base_delay: Base delay in seconds for exponential backoff (default: 2)
        timeout: Timeout in seconds for each request (default: 90)
    """
    
    # Determine if this is a search query request
    is_search_query = any(role['role'] == 'system' and 'search query generator' in role['content'].lower() 
                         for role in messages)
    
    # Since o3-mini is already fast, we don't need different configurations for fast_mode
    
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                delay = base_delay * (2 ** (attempt - 1))
                root_logger.info(f"Retry attempt {attempt + 1}/{max_retries} after {delay}s delay")
                time.sleep(delay)

            # Clean special characters from input messages
            cleaned_messages = []
            for msg in messages:
                cleaned_msg = msg.copy()
                cleaned_msg['content'] = clean_special_chars(msg['content'])
                cleaned_messages.append(cleaned_msg)

            print("\n=== O3-MINI INPUT ===")
            print(json.dumps(cleaned_messages, indent=2))

            # Create OpenAI client configuration
            client = OpenAI(
                api_key=os.environ["OPENAI_API_KEY"],
                base_url="https://api.openai.com/v1",
                http_client=httpx.Client(
                    trust_env=False,  # Disable environment-based proxy settings
                    timeout=httpx.Timeout(
                        timeout,  # Total timeout
                        connect=timeout,  # Connection timeout
                        read=timeout,  # Read timeout
                        write=timeout  # Write timeout
                    )
                ),
                max_retries=0  # We handle retries ourselves
            )

            response = client.chat.completions.create(
                model=model,
                messages=cleaned_messages,
                max_completion_tokens=8192  # Keep only the essential parameters
            )

            # Try to parse the response
            try:
                content = response.choices[0].message.content
            except (AttributeError, IndexError, json.JSONDecodeError) as e:
                root_logger.error(f"Failed to parse response: {str(e)}", exc_info=False)
                root_logger.debug(f"Raw response: {response}", exc_info=False)
                if attempt < max_retries - 1:
                    continue
                return "[Error: Failed to parse response]"

            # Clean special characters from output
            content = clean_special_chars(content.strip())
            
            # For search queries, extract just the first non-empty line
            if is_search_query:
                content = next((line.strip() for line in content.split('\n') 
                                if line.strip() and not line.strip().startswith('```')), content)

            # Sanitize and validate the output
            content = sanitize_json_content(content)
            if not validate_response(content, is_search_query=is_search_query):
                if attempt < max_retries - 1:
                    root_logger.warning("Response validation failed, retrying...", exc_info=False)
                    continue
                return "[Error: Response validation failed]"

            print("\n=== O3-MINI OUTPUT ===")
            print(content)
            print("=====================\n")

            return content

        except httpx.TimeoutException:
            error_msg = f"O3-mini API timeout after {timeout}s"
            root_logger.error(f"{error_msg} (attempt {attempt + 1}/{max_retries})", exc_info=False)
            if attempt < max_retries - 1:
                continue
            return f"[Error: {error_msg}]"

        except openai.APIError as e:
            error_msg = f"O3-mini API error: {str(e)}"
            root_logger.error(f"{error_msg} (attempt {attempt + 1}/{max_retries})", exc_info=False)
            if attempt < max_retries - 1:
                continue
            return f"[Error: {error_msg}]"

        except openai.RateLimitError as e:
            error_msg = f"O3-mini rate limit exceeded: {str(e)}"
            root_logger.error(f"{error_msg} (attempt {attempt + 1}/{max_retries})", exc_info=False)
            if attempt < max_retries - 1:
                time.sleep(base_delay * 4)  # Longer delay for rate limits
                continue
            return f"[Error: {error_msg}]"

        except json.JSONDecodeError as e:
            error_msg = f"Failed to parse JSON response: {str(e)}"
            root_logger.error(f"{error_msg} (attempt {attempt + 1}/{max_retries})", exc_info=False)
            if attempt < max_retries - 1:
                continue
            return f"[Error: {error_msg}]"

        except Exception as e:
            error_msg = f"O3-mini API call failed: {str(e)}"
            root_logger.error(f"{error_msg} (attempt {attempt + 1}/{max_retries})", exc_info=False)
            if attempt < max_retries - 1:
                continue
            return f"[Error: {error_msg}]"

    return "[Error: All retry attempts failed]"

def call_gemini(messages, model=GEMINI_FLASH_THINKING_ID, temperature=0.1, fast_mode=False, max_retries=1, base_delay=2, timeout=90):
    """
    Gemini version of the caller with retry logic and response sanitization.
    
    Args:
        messages: List of message dictionaries (will be converted to appropriate Gemini format)
        model: Model identifier (default: GEMINI_FLASH_THINKING_ID)
        temperature: Temperature for generation (default: 0.1)
        fast_mode: Ignored for Gemini
        max_retries: Maximum number of retry attempts (default: 1)
        base_delay: Base delay in seconds for exponential backoff (default: 2)
        timeout: Currently ignored for Gemini as it uses its own timeout handling
    """
    
    # Convert messages to Gemini format
    # Combine all messages into a single prompt, maintaining conversation structure
    prompt_parts = []
    for msg in messages:
        role = msg["role"].upper()
        content = msg["content"]
        prompt_parts.append(f"{role}: {content}")
    
    prompt = "\n".join(prompt_parts)
    
    print("\n=== GEMINI INPUT ===")
    print(prompt)
    
    for attempt in range(max_retries):
        try:
            if attempt > 0:
                delay = base_delay * (2 ** (attempt - 1))
                root_logger.info(f"Retry attempt {attempt + 1}/{max_retries} after {delay}s delay")
                time.sleep(delay)

            response = gemini_client.models.generate_content(
                model=model,
                contents=prompt,
                config=GenerateContentConfig(
                    tools=[],  # no tools => no searching
                    response_modalities=["TEXT"],
                    temperature=temperature
                )
            )
            
            # Extract text from response
            content = "".join(part.text for part in response.candidates[0].content.parts)
            content = content.strip()
            
            # Sanitize and validate the output
            content = sanitize_json_content(content)
            if not validate_response(content):
                if attempt < max_retries - 1:
                    root_logger.warning("Response validation failed, retrying...")
                    continue
                return "[Error: Response validation failed]"

            print("\n=== GEMINI OUTPUT ===")
            print(content)
            print("=====================\n")

            return content

        except Exception as e:
            error_msg = f"Gemini API call failed: {str(e)}"
            root_logger.error(f"{error_msg} (attempt {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                continue
            return f"[Error: {error_msg}]"

    return "[Error: All retry attempts failed]"


def call_deepseek_original(messages, model="deepseek-reasoner", temperature=0.1, fast_mode=False, max_retries=1, base_delay=2, timeout=90):
    """
    Original DeepSeek caller with retry logic, timeout, model fallback, and response sanitization.
    """
    # Override model choice if fast_mode is enabled
    if fast_mode:
        model = "deepseek-chat"
    
    original_model = model  # Store original model choice for potential fallback

    for attempt in range(max_retries):
        try:
            if attempt > 0:
                delay = base_delay * (2 ** (attempt - 1))
                root_logger.info(f"Retry attempt {attempt + 1}/{max_retries} after {delay}s delay")
                
                # If we've failed twice with reasoner, switch to chat model
                if attempt >= 2 and original_model == "deepseek-reasoner":
                    model = "deepseek-chat"
                    root_logger.info("Switching to faster chat model after repeated failures")
                
                time.sleep(delay)

            print("\n=== DEEPSEEK INPUT ===")
            print(json.dumps(messages, indent=2))

            # Create a clean configuration for Fireworks API
            client = openai.OpenAI(
                api_key=FIREWORKS_API_KEY,
                base_url="https://api.fireworks.ai/inference/v1",
                http_client=httpx.Client(
                    trust_env=False,
                    timeout=httpx.Timeout(
                        timeout,
                        connect=timeout,
                        read=timeout,
                        write=timeout
                    )
                ),
                max_retries=0
            )

            # Map DeepSeek model names to Fireworks model path
            model_map = {
                "deepseek-reasoner": "accounts/fireworks/models/deepseek-r1",
                "deepseek-chat": "accounts/fireworks/models/deepseek-r1"
            }
            fireworks_model = model_map.get(model, "accounts/fireworks/models/deepseek-r1")

            response = client.chat.completions.create(
                model=fireworks_model,
                messages=messages,
                temperature=temperature,
                max_tokens=20480,
                top_p=1,
                presence_penalty=0,
                frequency_penalty=0
            )

            try:
                content = response.choices[0].message.content
            except (AttributeError, IndexError, json.JSONDecodeError) as e:
                root_logger.error(f"Failed to parse response: {str(e)}")
                root_logger.debug(f"Raw response: {response}")
                if attempt < max_retries - 1:
                    continue
                return "[Error: Failed to parse response]"

            content = sanitize_json_content(content)
            if not validate_response(content):
                if attempt < max_retries - 1:
                    root_logger.warning("Response validation failed, retrying...")
                    continue
                return "[Error: Response validation failed]"

            print("\n=== DEEPSEEK OUTPUT ===")
            print(content)
            print("=====================\n")

            return content

        except httpx.TimeoutException:
            error_msg = f"DeepSeek API timeout after {timeout}s"
            root_logger.error(f"{error_msg} (attempt {attempt + 1}/{max_retries})")
            if model == "deepseek-reasoner":
                model = "deepseek-chat"
                root_logger.info("Timeout with reasoner model, switching to chat model")
            if attempt < max_retries - 1:
                continue
            return f"[Error: {error_msg}]"

        except openai.APIError as e:
            error_msg = f"DeepSeek API error: {str(e)}"
            root_logger.error(f"{error_msg} (attempt {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                continue
            return f"[Error: {error_msg}]"

        except openai.RateLimitError as e:
            error_msg = f"DeepSeek rate limit exceeded: {str(e)}"
            root_logger.error(f"{error_msg} (attempt {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                time.sleep(base_delay * 4)
                continue
            return f"[Error: {error_msg}]"

        except json.JSONDecodeError as e:
            error_msg = f"Failed to parse JSON response: {str(e)}"
            root_logger.error(f"{error_msg} (attempt {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                if model == "deepseek-reasoner":
                    model = "deepseek-chat"
                    root_logger.info("JSON parse error with reasoner model, switching to chat model")
                continue
            return f"[Error: {error_msg}]"

        except Exception as e:
            error_msg = f"DeepSeek API call failed: {str(e)}"
            root_logger.error(f"{error_msg} (attempt {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                continue
            return f"[Error: {error_msg}]"

    return "[Error: All retry attempts failed]"

# Main interface function that routes to the appropriate implementation
def call_deepseek(*args, **kwargs):
    """
    Main interface that routes to the chosen model implementation based on PREFERRED_MODEL setting.
    You can also override via environment variable MODEL_CHOICE=o3mini|gemini|deepseek
    """
    # Check environment variable first, then fall back to PREFERRED_MODEL
    model_choice = os.getenv("MODEL_CHOICE", PREFERRED_MODEL.value).lower()
    
    if model_choice == ModelChoice.GEMINI.value:
        return call_gemini(*args, **kwargs)
    elif model_choice == ModelChoice.DEEPSEEK.value:
        return call_deepseek_original(*args, **kwargs)
    else:  # Default to O3MINI
        return call_o3mini(*args, **kwargs)


def project_manager_agent(context, question, max_retries=3):
    """
    Research Question Analyst with retry logic.
    Uses the slower but more thorough reasoner model by default.
    """
    messages = [
        {"role": "system", "content": (
            "You are a Research Question Analyst. Your role is to:\n"
            "1. Break down complex research questions into their core components\n"
            "2. Identify key concepts and relationships to be investigated\n"
            "3. Highlight potential challenges in answering the question\n"
            "4. Suggest specific types of evidence needed\n"
            "5. Outline a clear research strategy\n"
            "6. Identify potential biases or limitations to consider"
        )},
        {"role": "user", "content": (
            f"Analyze this research question and provide guidance on how to best understand and answer it:\n\n"
            f"{question}\n\n"
            "Provide a structured analysis:\n"
            "1. CORE COMPONENTS:\n"
            "   - List main concepts and variables\n"
            "   - Identify relationships to investigate\n\n"
            "2. EVIDENCE NEEDED:\n"
            "   - Types of sources required\n"
            "   - Specific data or findings to look for\n\n"
            "3. RESEARCH STRATEGY:\n"
            "   - Suggested order of investigation\n"
            "   - Key sub-questions to focus on\n\n"
            "4. POTENTIAL CHALLENGES:\n"
            "   - Anticipated difficulties\n"
            "   - Possible limitations\n\n"
            "5. QUALITY CRITERIA:\n"
            "   - What makes a source relevant\n"
            "   - How to evaluate evidence quality"
        )}
    ]
    # Use reasoner model for project manager (no fast_mode)
    return call_deepseek(messages, max_retries=max_retries)


class SearchHistory:
    """Track search history to avoid duplicates and guide query generation"""
    def __init__(self):
        self.searches = []  # Complete search records
        self.used_queries = set()  # Exact queries already tried
        self.used_keywords = set()  # Individual keywords from all queries
        
    def add_search(self, query: str, results: list, was_successful: bool, raw_result: dict = None):  # Added raw_result parameter
        search_record = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "num_results": len(results),
            "was_successful": was_successful,
            "results_summary": self._summarize_results(results),
            "raw_result": raw_result  # Store the complete raw result
        }
        self.searches.append(search_record)
        self.used_queries.add(query.lower())
        
        # Extract and track keywords
        keywords = set(query.lower().split())
        self.used_keywords.update(keywords)
    
    def _summarize_results(self, results: list) -> list:
        """Create a concise summary of results"""
        summary = []
        for result in results:
            if isinstance(result, dict):
                summary.append({
                    "title": result.get("title", ""),
                    "snippet": result.get("snippet", "")[:2000],  # Truncate long snippets
                    "url": result.get("url", "")
                })
        return summary
    
    def get_search_history_prompt(self) -> str:
        """Generate a prompt summarizing search history for the LLM"""
        prompt = "Previous searches and results:\n\n"
        for search in self.searches:
            prompt += f"Query: {search['query']}\n"
            prompt += f"Results found: {search['num_results']}\n"
            if search['results_summary']:
                prompt += "Sample results:\n"
                for result in search['results_summary'][:2]:  # Show only first 2 results
                    prompt += f"- {result['title']}\n"
            prompt += "\n"
        
        prompt += "\nPreviously used keywords: " + ", ".join(sorted(self.used_keywords))
        return prompt



def researcher_agent(context, question, manager_plan=None):
    """
    Generates initial broad academic search queries, avoiding over-specificity while maintaining academic rigor.
    Balances between capturing core concepts and keeping sufficient breadth for comprehensive coverage.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a search query generator specialized in academic research. Follow these rules:\n"
                "1. Generate ONE academically-oriented search query with 1-5 keywords that:\n"
                "   - Uses academic/scientific terminology at an appropriate breadth\n"
                "   - Captures core concepts without being overly specific\n"
                "   - Includes field-specific terms only when essential\n"
                "   - Maintains sufficient breadth for comprehensive coverage\n\n"
                "2. Query requirements:\n"
                "   - NO Boolean operators (AND, OR, NOT)\n"
                "   - NO quotes or special characters\n"
                "   - Balance between precision and breadth\n"
                "   - NO explanatory text or formatting\n"
                "   - ONLY output the search terms\n\n"
                "3. Initial Search Strategy:\n"
                "   - Start with broader academic concepts\n"
                "   - Avoid immediate use of specialized scales or measures\n"
                "   - Focus on established terminology in the field\n"
                "   - Allow for discovery of related research\n\n"
                "CRITICAL: Return ONLY the simple keywords separated by spaces, nothing else.\n"
                "Example good output: psychological resilience health outcomes\n"
                "Example bad output: **Search Terms**: specific_scale_name anxiety measure"
            )
        },
        {
            "role": "user",
            "content": (
                f"Context:\n{context}\n\n"
                f"Research Plan:\n{manager_plan if manager_plan else 'Not provided'}\n\n"
                f"Create one broader academic search query for:\n{question}\n\n"
                "Requirements:\n"
                "1. Use academic/scientific terminology\n"
                "2. Keep sufficient breadth for comprehensive coverage\n"
                "3. Include field-specific terms only if essential\n"
                "4. Return ONLY the search terms, no other text\n"
                "5. Avoid over-specific or niche terminology in this initial query"
            )
        }
    ]
    
    response = call_deepseek(messages, fast_mode=True)
    
    # Clean the response more thoroughly
    # Remove markdown code blocks
    cleaned_response = re.sub(r'```.*?```', '', response, flags=re.DOTALL)
    # Remove <think> blocks
    cleaned_response = re.sub(r'<think>.*?</think>', '', cleaned_response, flags=re.DOTALL)
    # Remove any remaining markdown formatting
    cleaned_response = re.sub(r'[*_`#]', '', cleaned_response)
    # Get first non-empty line that's not just punctuation or whitespace
    for line in cleaned_response.split('\n'):
        line = line.strip()
        if line and not re.match(r'^[\s\W]+$', line):
            cleaned_response = line
            break
    
    # Log the generated query for debugging
    print("\n=== INITIAL (BROAD) SEARCH QUERY GENERATED ===")
    print(f"Raw response: {response}")
    print(f"Cleaned query: {cleaned_response}")
    print("=" * 50)
    
    return cleaned_response

def consecutive_researcher_agent(
    manager_plan: str, 
    user_question: str, 
    picked_results_so_far: List[dict],
    search_history: SearchHistory
) -> str:
    """
    Generate next search query based on history and previous results.
    Adaptively adjusts breadth based on current results and maintains comprehensive coverage.
    """
    reference_count = len(picked_results_so_far)
    
    messages = [
        {
            "role": "system",
            "content": (
                "You are a consecutive research query generator specialized in academic research. Your primary goal is to ensure "
                "comprehensive coverage while adapting to search success. Follow these rules:\n\n"
                "1. SEARCH STRATEGY - Alternate between THREE types of searches:\n"
                "   A. FOUNDATIONAL SEARCHES (broader terms):\n"
                "      - Use general academic terms that capture core concepts\n"
                "      - Focus on established relationships between key variables\n"
                "      - Prioritize when reference count is low or recent searches yield few results\n"
                "      - Example: 'psychological resilience health outcomes'\n\n"
                "   B. RELATIONSHIP SEARCHES (moderate specificity):\n"
                "      - Combine two core concepts\n"
                "      - Use alternative terminology for key relationships\n"
                "      - Focus on interaction effects and mechanisms\n"
                "      - Example: 'resilience moderation health outcomes'\n\n"
                "   C. SPECIFIC SEARCHES (precise terms):\n"
                "      - Include specific measures or scales only after foundational coverage\n"
                "      - Target exact mechanisms or relationships\n"
                "      - Use technical terminology for precise aspects\n"
                "      - Only when broader searches have yielded sufficient results\n\n"
                "2. ADAPTIVE STRATEGY:\n"
                "   - If references < 5: Use primarily foundational searches\n"
                "   - If recent searches yield few results: Return to broader terms\n"
                "   - If good foundation exists: Progress to more specific aspects\n\n"
                "3. QUERY REQUIREMENTS:\n"
                "   - Generate ONE search query with 1-5 keywords\n"
                "   - NO Boolean operators (AND, OR, NOT)\n"
                "   - NO quotes or special characters\n"
                "   - Avoid repeating exact previous combinations\n"
                "   - NO explanatory text or formatting\n\n"
                "4. COVERAGE CHECK:\n"
                "   - Ensure foundational literature is found for each key concept\n"
                "   - Look for baseline relationships before specific mechanisms\n"
                "   - Consider methodological and theoretical foundations\n"
                "   - Identify gaps in both broad and specific literature\n\n"
                "CRITICAL: Return ONLY the simple keywords separated by spaces, nothing else."
            )
        },
        {
            "role": "user",
            "content": (
                f"Manager's plan:\n{manager_plan}\n\n"
                f"User question:\n{user_question}\n\n"
                f"References found so far: {reference_count}\n"
                f"References details:\n{json.dumps(picked_results_so_far, indent=2)}\n\n"
                f"Search history:\n{search_history.get_search_history_prompt()}\n\n"
                "Generate ONE next search query following these priorities:\n"
                "1. If references < 5 or recent searches yielded few results: Use broader terms\n"
                "2. If foundational literature exists: Progress to more specific aspects\n"
                "3. If specific searches yield few results: Return to broader terms\n"
                "4. Ensure comprehensive coverage across concepts\n"
                "5. Consider methodological and theoretical foundations\n\n"
                "REMEMBER: Return ONLY the search terms, no other text or formatting"
            )
        }
    ]

    response = call_deepseek(messages, fast_mode=True)
    
    cleaned_response = re.sub(r'```.*?```', '', response, flags=re.DOTALL)
    cleaned_response = re.sub(r'<think>.*?</think>', '', cleaned_response, flags=re.DOTALL)
    cleaned_response = re.sub(r'[*_`#]', '', cleaned_response)
    
    for line in cleaned_response.split('\n'):
        line = line.strip()
        if line and not re.match(r'^[\s\W]+$', line):
            cleaned_response = line
            break
    
    print("\n=== CONSECUTIVE SEARCH QUERY GENERATED ===")
    print(f"Raw response: {response}")
    print(f"Cleaned query: {cleaned_response}")
    print("=" * 50)
    
    return cleaned_response



def content_developer_agent(context, question, manager_response, references_data, model_choice: Optional[str] = None):
    """
    Creates a comprehensive scientific text with citations.
    Now includes structured reference information and clear rationale for inclusion.
    Uses flatten_messages for more consistent prompt formatting.
    
    Args:
        context: Context information
        question: Research question to address
        manager_response: Project manager's analysis
        references_data: Available references and their metadata
        model_choice: Optional model override ('gemini', 'o3mini', 'deepseek', or None for default)
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a Knowledge Integration Specialist working with provided references and their metadata. "
                "Follow these rules:\n"
                "1. Every factual claim MUST have a complete scientific citation.\n"
                "2. Use APA style citations: (Author et al., year) in text, with full references at the end.\n"
                "3. Only cite sources that DIRECTLY support the claim based on their provided metadata/content.\n"
                "4. Use the full metadata from references (authors, year, title, journal, etc.).\n"
                "5. If a claim lacks direct support in the provided references, mark it as [needs citation].\n"
                "6. In the References list in the end, references must be in this format:\n"
                "   Author et al. (Year). Title. Journal. URL\n"
                "7. Stay within the scope of what's clearly stated in the provided references.\n"
                "8. Consider the inclusion rationale when using each reference.\n"
                "9. Incorporate references that establish relevant baseline relationships, even if incomplete.\n"
                "10. Clearly identify gaps where certain measures or relationships are missing.\n\n"
                "IMPORTANT REMINDER:\n"
                "- Work only with the metadata and content provided\n"
                "- Extract and use all available metadata in your citations\n"
                "- If metadata is missing, use what's available\n"
                "- If a claim lacks direct support, mark it as [needs citation]\n"
                "- Base claims only on what's explicitly stated in the provided content\n"
                "- Consider both initial and leftover picks in your synthesis\n"
                "- Emphasize what's known vs. what needs further research"
            )
        },
        {
            "role": "user",
            "content": (
                "### RESEARCH QUESTION ###\n"
                f"{question}\n\n"
                "### PROJECT MANAGER'S ANALYSIS ###\n"
                f"{manager_response}\n\n"
                "### AVAILABLE REFERENCES ###\n"
                f"{json.dumps(references_data, indent=2)}\n\n"
                "---\n\n"
                "Instructions:\n"
                "1. Answer the research question comprehensively\n"
                "2. Support each claim with complete scientific citations\n"
                "3. Mark unsupported claims as [needs citation]\n"
                "4. Group related citations if multiple sources support one claim\n"
                "5. Only make claims directly supported by the references\n"
                "6. Note any limitations based on available information\n"
                "7. Consider the inclusion rationale when using each reference\n"
                "8. Highlight both established relationships and research gaps\n"
                "9. End with a complete References list\n\n"
                f"Remember to fully address: {question}"
            )
        }
    ]
    
    # Flatten messages into a single prompt
    flattened_prompt = flatten_messages(messages)
    
    # Wrap the flattened prompt in a single message
    final_message = [{"role": "user", "content": flattened_prompt}]
    
    # Define fallback order for each model
    fallback_map = {
        "gemini": ["deepseek", "o3mini"],
        "o3mini": ["gemini", "deepseek"],
        "deepseek": ["gemini", "o3mini"]
    }
    
    def try_model(model_name: str) -> Optional[str]:
        """Helper function to try a specific model"""
        try:
            if "gemini" in model_name.lower():
                response = call_gemini(final_message)
            elif "o3mini" in model_name.lower():
                response = call_o3mini(final_message)
            elif "deepseek" in model_name.lower():
                response = call_deepseek_original(final_message)
            else:
                response = call_deepseek(final_message)
                
            if response and not response.startswith("[Error"):
                return response
        except Exception as e:
            root_logger.error(f"{model_name} generation failed: {str(e)}")
        return None

    # Try primary model first
    model_choice_lower = (model_choice or "deepseek").lower()
    response = try_model(model_choice_lower)
    if response:
        return response
        
    # Try fallbacks if primary fails
    for fallback in fallback_map.get(model_choice_lower, []):
        print(f"\nTrying fallback model: {fallback}")
        response = try_model(fallback)
        if response:
            return response
            
    # If all attempts fail, use default DeepSeek as final fallback
    print("\nAll specified models failed, trying default DeepSeek...")
    return call_deepseek(final_message)



def search_result_picker_agent(
    user_question: str,
    raw_search_results: dict
) -> dict:
    """
    Filters raw search results and returns potentially relevant ones in a structured format.
    
    Enhanced to:
    1. Include foundational studies that establish baseline relationships
    2. Consider partial matches that address key components
    3. Identify research gaps through missing connections
    4. Store full metadata and inclusion rationale
    5. Parse responses robustly using extract_json
    """
    print("\n=== DEBUG: Starting search_result_picker_agent ===")
    
    if isinstance(raw_search_results, str):
        print(f"DEBUG: raw_search_results is a string, attempting to parse")
        try:
            raw_search_results = json.loads(raw_search_results)
            print("DEBUG: Successfully parsed raw_search_results to dict")
        except json.JSONDecodeError as e:
            print(f"Failed to parse raw_search_results string: {e}")
            return {"status": "ERROR", "relevant_results": []}

    messages = [
        {
            "role": "system",
            "content": (
                "You are a search result filter focused on building comprehensive evidence bases "
                "from ACADEMIC SOURCES ONLY. Your primary goal is to identify THREE types of valuable sources:\n\n"
                "1. FOUNDATIONAL EVIDENCE:\n"
                "   - Peer-reviewed studies establishing baseline relationships\n"
                "   - Academic research documenting core outcomes\n"
                "   - Important even if they don't address all aspects\n\n"
                "2. DIRECT MATCHES:\n"
                "   - Academic sources addressing multiple aspects\n"
                "   - Peer-reviewed studies combining key concepts\n"
                "   - Scholarly research with comprehensive coverage\n\n"
                "3. GAP-IDENTIFYING SOURCES:\n"
                "   - Academic studies highlighting missing connections\n"
                "   - Scholarly research suggesting future directions\n"
                "   - Peer-reviewed work partially addressing the question\n\n"
                "INCLUSION CRITERIA - Include ONLY if the source:\n"
                "- Is from a peer-reviewed academic journal\n"
                "- Is a scholarly conference proceeding\n"
                "- Is a research institute publication\n"
                "- Is an academic thesis or dissertation\n\n"
                "AUTOMATICALLY EXCLUDE:\n"
                "- News articles or media coverage\n"
                "- Commercial websites\n"
                "- Blog posts or opinion pieces\n"
                "- Practice updates or clinical newsletters\n"
                "- General information websites\n"
                "- Non-peer-reviewed sources\n\n"
                "Return ONLY a JSON object with this structure:\n"
                "{\n"
                "  \"relevant_urls\": [\"url1\", \"url2\"],\n"
                "  \"rationale\": {\n"
                "    \"url1\": \"Specific value (e.g., 'Peer-reviewed study on X')\",\n"
                "    \"url2\": \"Specific value (e.g., 'Academic research on Y')\"\n"
                "  }\n"
                "}"
            )
        },
        {
            "role": "user",
            "content": (
                f"User question:\n{user_question}\n\n"
                "Raw search results:\n"
                f"{json.dumps(raw_search_results, indent=2)}\n\n"
                "Evaluate these results, remembering to:\n"
                "1. Include foundational studies that establish key relationships\n"
                "2. Consider partial matches that address important components\n"
                "3. Value sources that help identify research gaps\n"
                "4. Provide specific rationale for each inclusion\n"
                "Return JSON with URLs and detailed rationale for each inclusion."
            )
        }
    ]

    try:
        response = call_deepseek(messages, fast_mode=True)
        print("\nDEBUG: DeepSeek Response:")
        print(response)

        parsed_jsons = extract_json(response)
        print(f"\nDEBUG: Extracted JSONs: {json.dumps(parsed_jsons, indent=2)}")

        if not parsed_jsons:
            print("DEBUG: No valid JSON found in the DeepSeek response.")
            return {"status": "NO_RELEVANT_SEARCH_RESULTS", "relevant_results": []}

        # Take the first valid JSON that has both 'relevant_urls' and 'rationale'
        relevant_urls = []
        rationale = {}
        for obj in parsed_jsons:
            if isinstance(obj, dict) and "relevant_urls" in obj:
                relevant_urls = obj["relevant_urls"]
                rationale = obj.get("rationale", {})
                break

        if not relevant_urls:
            print("DEBUG: No relevant URLs found in parsed JSON.")
            return {"status": "NO_RELEVANT_SEARCH_RESULTS", "relevant_results": []}

        # We have a list of relevant URLs. Let's match them with the raw_search_results.
        # raw_search_results may have a top-level "results" key or nested within "results.results".
        results_list = []
        if isinstance(raw_search_results, dict):
            # Some versions of Serper results
            if "results" in raw_search_results and isinstance(raw_search_results["results"], list):
                results_list = raw_search_results["results"]
            elif "results" in raw_search_results and isinstance(raw_search_results["results"], dict):
                possible_inner = raw_search_results["results"].get("results", [])
                if isinstance(possible_inner, list):
                    results_list = possible_inner

        # Normalize function to check URL matching
        def normalize_url(url):
            return url.strip().rstrip('/').replace('\\', '/').lower()

        normalized_urls = set(normalize_url(u) for u in relevant_urls)
        # Create a normalized rationale map
        normalized_rationale = {
            normalize_url(url): reason 
            for url, reason in rationale.items()
        }

        relevant_results = []
        excluded_results = []
        for entry in results_list:
            if not isinstance(entry, dict):
                continue
            entry_url = entry.get("url", "")
            normalized_entry_url = normalize_url(entry_url)
            if normalized_entry_url in normalized_urls:
                # Try both raw and normalized URLs for rationale
                entry["inclusion_rationale"] = (
                    rationale.get(entry_url) or 
                    normalized_rationale.get(normalized_entry_url) or 
                    "No specific rationale provided"
                )
                relevant_results.append(entry)
            else:
                excluded_results.append({
                    "url": entry_url,
                    "title": entry.get("title", "No title"),
                    "reason": "Not matched by DeepSeek relevance filter"
                })

        # Log excluded results
        if excluded_results:
            print("\nDEBUG: Excluded Results:")
            for result in excluded_results:
                print(f"- {result['title']}\n  URL: {result['url']}\n  Reason: {result['reason']}\n")

        if relevant_results:
            return {
                "status": "SUCCESS",
                "relevant_results": relevant_results,
                "excluded_results": excluded_results
            }
        else:
            print("DEBUG: No matching search results found among raw_search_results.")
            return {"status": "NO_RELEVANT_SEARCH_RESULTS", "relevant_results": []}

    except Exception as e:
        print(f"Unexpected error in search_result_picker_agent: {str(e)}")
        return {"status": "ERROR", "relevant_results": []}



def quality_reviewer_agent(context, question):
    """
    Reviews content quality with model fallbacks.
    """
    # Parse context if it's a string
    if isinstance(context, str):
        try:
            context = json.loads(context)
        except json.JSONDecodeError:
            context = {"content": context}  # Fallback if JSON parsing fails

    messages = [
        {
            "role": "system",
            "content": (
                "You are a Content Excellence Auditor focusing on citation accuracy and evidence quality. Check that:\n"
                "1. Each citation directly supports its specific claim\n"
                "2. No claims are made without proper citation support\n"
                "3. Citations contain actual evidence, not just mentions\n"
                "4. The evidence-to-claim relationship is clear and accurate\n"
                "5. Pay special attention to rejected citations and their impact\n"
                "6. Consider the overall citation quality metrics provided\n\n"
                "IMPORTANT GUIDELINES:\n"
                "- Never reject citations due to technical issues like fetch errors or partial metadata\n"
                "- Accept citations with partial metadata if content is relevant (use 'Unknown Author'/'n.d.' when needed)\n"
                "- Flag missing author/year but maintain citation if content supports claim\n"
                "- Use available metadata to reconstruct citations where possible\n"
                "- Focus evaluation on CONTENT RELEVANCE rather than technical completeness\n"
                "- Only recommend excluding citations if they are:\n"
                "  a) Completely irrelevant to the claims they support\n"
                "  b) Contain factually incorrect information\n"
                "  c) Are fundamentally flawed in methodology\n"
                "- Base evaluations on available evidence, even if limited\n"
                "- For PDFs with extraction issues: Check raw metadata before rejection"
            )
        },
        {
            "role": "user",
            "content": (
                "### ORIGINAL QUESTION ###\n"
                f"{question}\n\n"
                "### PROJECT PLAN ###\n"
                f"{context.get('plan', '')}\n\n"
                "### CONTENT TO REVIEW ###\n"
                f"{context.get('content', '')}\n\n"
                "### CITATION REVIEWS ###\n"
                f"{context.get('citation_reviews', '')}\n\n"
                "---\n\n"
                "Review the content focusing on:\n"
                "1. Citation accuracy and evidence quality\n"
                "2. Impact of rejected citations on conclusions\n"
                "3. Strength of evidence for key claims\n"
                "4. Overall reliability of sources\n\n"
                "Provide structured feedback:\n"
                "1. MAJOR ISSUES: List any critical problems with evidence or claims\n"
                "2. EVIDENCE GAPS: Identify claims needing better support\n"
                "3. SOURCE QUALITY: Comment on evidence quality and relevance\n"
                "4. RECOMMENDATIONS: Suggest specific improvements\n\n"
                "Remember:\n"
                "- Evaluate citations based on available evidence and metadata\n"
                "- Focus on evidence quality and claim support\n"
                f"- Assess how well the content answers: {question}"
            )
        }
    ]

    def try_model(model_func, messages):
        try:
            response = model_func(messages)
            if response and not response.startswith("[Error"):
                return response
        except Exception as e:
            root_logger.error(f"{model_func.__name__} failed: {str(e)}")
        return None

    # Try O3-mini first
    result = try_model(call_o3mini, messages)
    if result:
        return result

    # Fallback to Gemini
    print("\nFalling back to Gemini for quality review...")
    result = try_model(call_gemini, messages)
    if result:
        return result

    # Final fallback to DeepSeek
    print("\nFalling back to DeepSeek for quality review...")
    return call_deepseek_original(messages)


def leftover_references_evaluator(
    user_question: str,
    manager_plan: str,
    all_search_results: List[dict],
    picked_references: List[dict],
    draft_content: str  # New parameter
) -> List[dict]:
    """
    Evaluates references that weren't used in the initial draft, including both:
    - References that weren't initially picked
    - References that were picked but not used in the draft
    
    Args:
        user_question: Original research question
        manager_plan: Project manager's analysis and plan
        all_search_results: All search results from session_search_results_raw.json
        picked_references: References that were already picked
        draft_content: The content of the first draft to analyze for used references
    
    Returns:
        List of additional relevant references to include
    """
    # Debug logging at start
    root_logger.debug(f"Starting leftover evaluation with {len(all_search_results)} search results and {len(picked_references)} picked references")

    # Extract URLs from the draft content
    def extract_urls_from_draft(content: str) -> Set[str]:
        # Common URL patterns in academic citations and references
        url_patterns = [
            r'https?://[^\s<>"]+|www\.[^\s<>"]+',  # Basic URLs
            r'(?<=\()https?://[^\s<>")]+',         # URLs in parentheses
            r'(?<=\[)https?://[^\s<>\]]+',         # URLs in square brackets
        ]
        
        used_urls = set()
        for pattern in url_patterns:
            urls = re.findall(pattern, content)
            normalized_urls = {normalize_url(url) for url in urls}
            used_urls.update(normalized_urls)
            
        root_logger.debug(f"Found {len(used_urls)} URLs used in draft")
        return used_urls

    # Normalize function for consistent URL handling
    def normalize_url(url):
        return url.strip().rstrip('/').replace('\\', '/').lower()

    # Extract used URLs from draft
    used_urls = extract_urls_from_draft(draft_content)
    root_logger.info(f"Found {len(used_urls)} URLs used in the draft")

    # Create a dictionary of picked references for easier merging
    picked_refs_dict = {
        normalize_url(ref.get('url', '')): ref
        for ref in picked_references
        if ref.get('url')
    }
    picked_urls = set(picked_refs_dict.keys())
    
    root_logger.debug(f"Number of already picked URLs: {len(picked_urls)}")
    
    # Filter out already picked references with more permissive results extraction
    leftover_refs = []
    total_results_processed = 0
    
    # Handle both nested and flat result formats
    for result in all_search_results:
        if not isinstance(result, dict):
            continue
            
        # If result is already in the expected format (has url, title, etc.)
        if 'url' in result:
            total_results_processed += 1  # Count all results
            url = result.get('url', '')
            normalized_url = normalize_url(url)
            root_logger.debug(f"Checking URL: {url}")
            root_logger.debug(f"Normalized: {normalized_url}")
            
            if url and normalized_url not in used_urls:
                if normalized_url in picked_urls:
                    # Merge metadata with existing reference
                    existing_ref = picked_refs_dict[normalized_url]
                    # Update snippet if new one is longer
                    if len(result.get('snippet', '')) > len(existing_ref.get('snippet', '')):
                        existing_ref['snippet'] = result.get('snippet', '')
                    # Merge other metadata fields
                    for key, value in result.items():
                        if key not in existing_ref or (isinstance(value, str) and len(value) > len(existing_ref.get(key, ''))):
                            existing_ref[key] = value
                    root_logger.debug(f"Merged metadata for picked reference: {url}")
                else:
                    leftover_refs.append(result)
                    root_logger.debug(f"Found unused leftover reference: {url}")
                    
        # If result contains nested results
        elif 'results' in result:
            results = result['results']
            if isinstance(results, str):
                try:
                    results = json.loads(results)
                except json.JSONDecodeError:
                    continue
            
            # Handle both possible nested structures
            if isinstance(results, dict):
                results_list = results.get('results', [])
                if not results_list and 'results' in results:
                    results_list = results['results']
            elif isinstance(results, list):
                results_list = results
            else:
                continue
            
            total_results_processed += len(results_list)
            
            for entry in results_list:
                if isinstance(entry, dict):
                    url = entry.get('url', '')
                    normalized_url = normalize_url(url)
                    if url and normalized_url not in used_urls:
                        if normalized_url in picked_urls:
                            # Merge metadata with existing reference
                            existing_ref = picked_refs_dict[normalized_url]
                            if len(entry.get('snippet', '')) > len(existing_ref.get('snippet', '')):
                                existing_ref['snippet'] = entry.get('snippet', '')
                            for key, value in entry.items():
                                if key not in existing_ref or (isinstance(value, str) and len(value) > len(existing_ref.get(key, ''))):
                                    existing_ref[key] = value
                            root_logger.debug(f"Merged metadata for picked reference: {url}")
                        else:
                            leftover_refs.append(entry)
                            root_logger.debug(f"Found unused leftover reference: {url}")

    root_logger.info(f"Processed {total_results_processed} total results")
    root_logger.info(f"Found {len(leftover_refs)} leftover references to evaluate")
    
    # Add picked but unused references
    for ref in picked_references:
        url = ref.get('url', '')
        normalized_url = normalize_url(url)
        if url and normalized_url not in used_urls:  # Only add if not used in draft
            # Only add if not already in leftover_refs
            if not any(normalize_url(r.get('url', '')) == normalized_url for r in leftover_refs):
                leftover_refs.append(ref)
                root_logger.debug(f"Added unused picked reference: {url}")

    root_logger.info(f"Total references to evaluate (excluding used refs): {len(leftover_refs)}")
    
    if not leftover_refs:
        root_logger.debug("Search results structure (first 2 items):")
        root_logger.debug(json.dumps(all_search_results[:2], indent=2))
        root_logger.debug("Picked references structure (first 2 items):")
        root_logger.debug(json.dumps(picked_references[:2], indent=2))
        return []

    # Prepare concise version of references for the LLM
    concise_refs = [{
        'url': ref.get('url', ''),
        'title': ref.get('title', 'No title'),
        'snippet': ref.get('snippet', '')[:500] + ('...' if len(ref.get('snippet', '')) > 500 else '')
    } for ref in leftover_refs]

    messages = [
        {
            "role": "system",
            "content": (
                "You are a Reference Re-evaluation Specialist. Your task is to identify valuable references "
                "that might have been missed in the initial selection. Consider THREE types of valuable sources:\n\n"
                "1. FOUNDATIONAL EVIDENCE:\n"
                "   - Studies establishing baseline relationships between key variables\n"
                "   - Research documenting core outcomes or effects\n"
                "   - Important even if they don't address all aspects of the question\n\n"
                "2. DIRECT MATCHES:\n"
                "   - Sources that explicitly address multiple aspects of the research question\n"
                "   - Studies combining key concepts in relevant ways\n"
                "   - Research with comprehensive coverage of the topic\n\n"
                "3. GAP-IDENTIFYING SOURCES:\n"
                "   - Studies that highlight missing connections\n"
                "   - Research that suggests needed future directions\n"
                "   - Work that partially addresses the question\n\n"
                "INCLUSION CRITERIA - Include if the source:\n"
                "- Provides foundational evidence for ANY key relationship\n"
                "- Contains relevant data, methods, or frameworks\n"
                "- Offers expert insights or professional perspectives\n"
                "- Helps identify research gaps\n"
                "- Establishes baseline relationships\n\n"
                "CRITICAL: If a reference establishes a baseline relationship or partial link "
                "that was previously overlooked, prioritize re-including it.\n\n"
                "Return ONLY a JSON object with this structure:\n"
                "{\n"
                "  \"additional_urls\": [\"url1\", \"url2\"],\n"
                "  \"rationale\": {\"url1\": \"reason for inclusion\", \"url2\": \"reason for inclusion\"}\n"
                "}"
            )
        },
        {
            "role": "user",
            "content": (
                f"Research Question:\n{user_question}\n\n"
                f"Research Plan:\n{manager_plan}\n\n"
                f"Total references to evaluate: {len(leftover_refs)}\n"
                f"Already picked references: {len(picked_references)}\n\n"
                "Evaluate these leftover references for potential inclusion:\n"
                f"{json.dumps(concise_refs, indent=2)}\n\n"
                "Consider references that could:\n"
                "1. Strengthen evidence for relationships between variables\n"
                "2. Add valuable context or background\n"
                "3. Support methodological decisions\n"
                "4. Provide important insights\n"
                "Return JSON with URLs to include and rationale for each."
            )
        }
    ]

    try:
        response = call_deepseek(messages, fast_mode=True)
        root_logger.debug(f"\nDEBUG: DeepSeek Response:\n{response}")
        parsed_jsons = extract_json(response)
        root_logger.debug(f"\nDEBUG: Extracted JSONs: {json.dumps(parsed_jsons, indent=2)}")
        
        if not parsed_jsons:
            root_logger.warning("No valid JSON found in leftover references evaluation")
            return []
            
        # Use the first valid JSON that has the expected structure
        for result in parsed_jsons:
            if isinstance(result, dict) and "additional_urls" in result:
                additional_urls = result["additional_urls"]
                rationale = result.get("rationale", {})
                
                # Create normalized rationale map
                normalized_rationale = {
                    normalize_url(url): reason 
                    for url, reason in rationale.items()
                }
                
                # Log the decisions
                root_logger.info("\n=== LEFTOVER REFERENCES EVALUATION ===")
                for url in additional_urls:
                    reason = rationale.get(url, "No specific rationale provided")
                    root_logger.info(f"\nIncluding: {url}\nReason: {reason}")
                
                # Return the full reference objects for the additional URLs
                additional_refs = []
                for ref in leftover_refs:
                    url = ref.get('url', '')
                    normalized_url = normalize_url(url)
                    if url in additional_urls or normalized_url in {normalize_url(u) for u in additional_urls}:
                        ref['inclusion_rationale'] = (
                            rationale.get(url) or 
                            normalized_rationale.get(normalized_url) or 
                            "No specific rationale provided"
                        )
                        additional_refs.append(ref)
                
                return additional_refs
                
        return []
        
    except Exception as e:
        root_logger.error(f"Error in leftover references evaluation: {str(e)}")
        return []


def final_revision_agent(context, question):
    """
    Creates or revises a comprehensive scientific text with citations.
    Works with the provided context and citation metadata to:
    1. Use the full metadata from references to understand the source
    2. Include both direct evidence and foundational/baseline studies
    3. Use proper citation format with all available metadata
    4. Note any limitations based on the available information
    """
    # Parse context if it's a string
    if isinstance(context, str):
        try:
            context = json.loads(context)
        except json.JSONDecodeError:
            context = {"content": context}  # Fallback if JSON parsing fails

    # Extract original draft if available
    original_draft = context.get("original_draft", "")
    current_text = context.get("current_text", context.get("content", ""))

    messages = [
        {
            "role": "system",
            "content": (
                "You are a Knowledge Integration Specialist working with provided references and their metadata. "
                "Treat reviewer comments as valuable co-author input that can enhance the manuscript.\n\n"
                "Follow these rules:\n"
                "1. Every factual claim MUST have a complete scientific citation.\n"
                "2. Use APA style citations: (Author et al., year) in text, with full references at the end.\n"
                "3. Value both direct and foundational evidence:\n"
                "   - Direct evidence: Sources that explicitly address specific claims\n"
                "   - Foundational evidence: Sources that establish important baseline relationships\n"
                "   - Contextual evidence: Sources that provide valuable background or methodology\n"
                "4. Use the full metadata from references (authors, year, title, journal, etc.).\n"
                "5. Actively incorporate reviewer insights to enhance the text:\n"
                "   - Use their suggested nuances and clarifications\n"
                "   - Integrate their methodological observations\n"
                "   - Include their identified research gaps as valuable context\n"
                "   - Adopt their suggestions for more precise language\n"
                "6. For topics lacking direct evidence:\n"
                "   - Use foundational studies to establish baseline relationships\n"
                "   - Discuss partial evidence and its implications\n"
                "   - Note where more specific research is needed\n"
                "   - Clearly distinguish known relationships from hypothesized ones\n"
                "7. Build evidence progressively:\n"
                "   - Start with foundational/baseline relationships\n"
                "   - Add specific evidence where available\n"
                "   - Note where relationships are established vs. hypothesized\n"
                "   - Explicitly discuss which variables are measured vs. missing\n"
                "8. Structure paragraphs to build understanding:\n"
                "   - Begin with established baseline relationships\n"
                "   - Layer on specific findings where available\n"
                "   - Clearly identify gaps and needed research\n"
                "   - Use foundational studies even if they lack certain measures\n"
                "9. In the References list, use format: (Author et al., Year, Title, Journal, URL)\n\n"
                "IMPORTANT REMINDER:\n"
                "- Retain references that provide foundational or contextual evidence\n"
                "- Include baseline studies even if they don't address every variable\n"
                "- Use partial evidence to build toward fuller understanding\n"
                "- Transform technical review comments into reader-friendly insights\n"
                "- Integrate limitations and evidence gaps naturally into the narrative\n"
                "- Preserve all domain-specific terminology and references from the original draft"
            )
        },
        {
            "role": "user",
            "content": (
                "### ORIGINAL QUESTION ###\n"
                f"{question}\n\n"
                "### ORIGINAL DRAFT ###\n"
                f"{original_draft}\n\n"
                "### CURRENT TEXT ###\n"
                f"{current_text}\n\n"
                "### CITATION REVIEWS ###\n"
                f"{context.get('citation_reviews', '')}\n\n"
                "### QUALITY REVIEW ###\n"
                f"{context.get('quality_review', '')}\n\n"
                "---\n\n"
                "Instructions:\n"
                "1. Create a revised version that addresses ALL feedback from both reviews\n"
                "2. Support each claim with complete scientific APA style citations\n"
                "3. Remove or revise claims with rejected citations\n"
                "4. Mark unsupported claims as [needs citation]\n"
                "5. Group related citations if multiple sources support one claim\n"
                "6. Only make claims that are directly supported by the provided references\n"
                "7. Note any limitations based on the available information\n"
                "8. End with a complete 'References:' list\n"
                "9. Preserve all domain-specific terminology and references\n\n"
                f"Remember to fully address: {question}\n\n"
            )
        }
    ]
    
    def try_model(model_func, messages, attempt):
        """Helper function to try a specific model"""
        try:
            response = model_func(messages)
            if response and not response.startswith("[Error"):
                return response
        except Exception as e:
            root_logger.error(f"{model_func.__name__} attempt {attempt + 1} failed: {str(e)}")
        return None

    # Add retry logic with model fallbacks
    max_retries = 3
    for attempt in range(max_retries):
        # Try DeepSeek first
        result = try_model(call_deepseek, messages, attempt)
        if result:
            return result
            
        # Fallback to O3-mini
        print("\nFalling back to O3-mini for final revision...")
        result = try_model(call_o3mini, messages, attempt)
        if result:
            return result
            
        # Fallback to Gemini
        print("\nFalling back to Gemini for final revision...")
        result = try_model(call_gemini, messages, attempt)
        if result:
            return result
            
        # If all models fail, wait before retry
        if attempt < max_retries - 1:
            wait_time = 2 ** attempt  # Exponential backoff
            print(f"\nAll models failed, waiting {wait_time}s before retry...")
            time.sleep(wait_time)
        else:
            root_logger.error(f"Final revision failed after {max_retries} attempts with all models")
            return current_text  # Fallback to current version


def single_citation_check(call_deepseek, url, full_text, fallback_metadata, question, draft_context):
    """
    Enhanced citation relevance checker with three-way evaluation criteria.
    In addition to checking whether the given citation (by URL) supports the draft, this function
    instructs the model to scan the full source content for any extra citations that might be useful.
    
    The expected output should include:
      - A decision for the provided citation: ACCEPT, REVISE_USE, or REJECT.
      - A JSON array under the key "extra_citations" listing any additional citations
        (with full citation text and reviewer instructions) if present.
    """
    max_retries = 3
    base_timeout = 60  # Base timeout for Gemini
    
    # Convert fallback metadata to a readable JSON string if it's a dict
    if isinstance(fallback_metadata, dict):
        fallback_metadata_str = json.dumps(fallback_metadata, indent=2)
    else:
        fallback_metadata_str = str(fallback_metadata)

    user_message_content = (
        "You are a Citation Usage Reviewer specializing in academic literature. Your task is to evaluate citations "
        "considering THREE types of valid evidence, focusing exclusively on peer-reviewed academic sources:\n\n"
        "1. DIRECT EVIDENCE from academic literature:\n"
        "   - Peer-reviewed research findings\n"
        "   - Primary research data and analysis\n"
        "   - Systematic reviews or meta-analyses\n\n"
        "2. FOUNDATIONAL EVIDENCE from scholarly sources:\n"
        "   - Theoretical frameworks from academic publications\n"
        "   - Established scientific principles\n"
        "   - Peer-reviewed methodological foundations\n\n"
        "3. CONTEXTUAL EVIDENCE from academic sources:\n"
        "   - Research methodologies from scholarly papers\n"
        "   - Academic background literature\n"
        "   - Scholarly field conventions\n\n"
        "EVALUATION CRITERIA:\n"
        "- ACCEPT: Academic source provides valid direct, foundational, or contextual evidence\n"
        "- REVISE_USE: Academic source is valuable but needs clearer connection to claims\n"
        "- REJECT: Source is non-academic, irrelevant, or fundamentally flawed\n\n"
        "AUTOMATICALLY REJECT:\n"
        "- News articles or press releases\n"
        "- Commercial or corporate websites\n"
        "- Blog posts or opinion pieces\n"
        "- Non-peer-reviewed sources\n"
        "- Practice updates or clinical newsletters\n"
        "- General information websites\n\n"
        f"Research Question: {question}\n\n"
        f"Draft Text Context: {draft_context}\n\n"
        f"URL of the citation under review: {url}\n\n"
        "=== SOURCE CONTENT (TRUNCATED IF LONG) ===\n"
        f"{full_text[:100000] if full_text else '[No full text available]'}\n\n"
        "=== ADDITIONAL METADATA ===\n"
        f"{fallback_metadata_str if fallback_metadata_str else '[No metadata available]'}\n\n"
        "=== FINAL INSTRUCTIONS - IMPORTANT ===\n"
        "Remember the three types of valid evidence to consider:\n"
        "1. DIRECT: Explicitly addresses claims with specific findings\n"
        "2. FOUNDATIONAL: Establishes baseline relationships or framework\n"
        "3. CONTEXTUAL: Provides methodology or background\n\n"
        "Your evaluation criteria:\n"
        "- ACCEPT: Valid direct, foundational, or contextual evidence\n"
        "- REVISE_USE: Valuable but needs clearer connection\n"
        "- REJECT: Irrelevant or fundamentally flawed\n\n"
        "2. EXTRA CITATIONS (if any): Scan the full source content for any additional citations that are mentioned or referenced but not currently used in the draft.\n"
        "   - For each extra citation, provide the full citation as it appears in the references list in the source and a brief instruction on how the reviewer might consider incorporating it.\n"
        "   - Return these extra citations as a JSON array under the key 'extra_citations'. If no extra citations are found, return an empty array.\n\n"
        "Focus on both the accuracy of the citation usage for the given URL and on identifying any additional, potentially useful citations."
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a Citation Usage Reviewer: verify if the provided citation supports the claims in the draft and "
                "scan the source content for any extra citations that might be useful. Your output should include your decision "
                "for the provided citation and, if applicable, a JSON array of extra citations under the key 'extra_citations'."
            )
        },
        {
            "role": "user",
            "content": user_message_content
        }
    ]

    for attempt in range(max_retries):
        try:
            # Use call_gemini instead of call_deepseek for higher token capacity
            timeout = base_timeout * (attempt + 1)
            root_logger.info(f"Citation check attempt {attempt + 1}/{max_retries} for {url} with {timeout}s timeout")
            
            response = call_gemini(messages, timeout=timeout)
            if response and not response.startswith("[Error"):
                return response
                
            if "timeout" in response.lower():
                root_logger.warning(f"Timeout on attempt {attempt + 1}, retrying with increased timeout...")
                continue
                
            root_logger.error(f"Gemini error on attempt {attempt + 1}: {response}")
            
        except Exception as e:
            root_logger.error(f"Citation check attempt {attempt + 1} failed: {str(e)}")
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                root_logger.info(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            else:
                return f"[Error: Citation check failed after {max_retries} attempts: {str(e)}]"

    return f"[Error: Citation check failed after {max_retries} attempts with timeouts]"





def is_valid_url(url: str) -> bool:
    """Validate URL format"""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

def clean_text(text: str) -> str:
    """Sanitize text content and remove noisy symbols"""
    text = unescape(text).replace('\x00', '').strip()

    # Remove sequences of non-alphanumeric characters
    text = re.sub(r'[^a-zA-Z0-9\s]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip() # Normalize whitespace again after symbol removal
    return text


def process_pdf(pdf_file: BytesIO) -> str:
    """Improved PDF text extraction with metadata fallback"""
    text = ""
    try:
        # First try PyMuPDF for better text extraction
        import fitz  # PyMuPDF
        doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        
        # Extract text page by page with proper spacing
        pages = []
        for page in doc:
            page_text = page.get_text("text", flags=fitz.TEXT_PRESERVE_WHITESPACE)
            if page_text.strip():
                pages.append(page_text)
        
        text = "\n\n".join(pages)
        
        # If PyMuPDF fails to extract meaningful text, try pypdf
        if not text.strip():
            pdf_file.seek(0)
            reader = PdfReader(pdf_file)
            pages = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text.strip():
                    pages.append(page_text)
            text = "\n\n".join(pages)
        
        return clean_text(text)
        
    except ImportError:
        # Fallback to pypdf
        try:
            reader = PdfReader(pdf_file)
            return clean_text(' '.join([page.extract_text() for page in reader.pages]))
        except Exception as e:
            print(f"PDF processing failed: {str(e)}")
            return ""
    except Exception as e:
        print(f"PDF processing failed: {str(e)}")
        return ""



async def check_citations_in_content_async(content: str, search_results: list, user_query: str, call_deepseek) -> Dict:
    """
    Asynchronously check citations with controlled concurrency and result aggregation.
    Now uses the full metadata from the references when fetch fails or is partial.
    """
    results = {
        "citations": {},
        "summary": {
            "total_urls": 0,
            "successful_fetches": 0,
            "failed_fetches": 0
        }
    }
    
    # Extract URLs and create search result mapping
    urls = set(url for url in re.findall(
        r'https?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', content
    ) if is_valid_url(url))
    
    # Create a map of URL to the associated reference object
    url_search_map = create_url_search_map(search_results)
    
    if not urls:
        root_logger.warning("No valid URLs found in content")
        return results
    
    results["summary"]["total_urls"] = len(urls)
    root_logger.info(f"Found {len(urls)} URLs to check")
    
    # Set up async session and semaphore for rate limiting
    async with aiohttp.ClientSession() as session:
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        
        # Create tasks for all URLs
        fetch_tasks = [
            fetch_full_text_async(url, session, semaphore)
            for url in urls
        ]
        
        # Execute all fetch tasks concurrently
        fetch_results = await asyncio.gather(*fetch_tasks, return_exceptions=True)
        
        # Process results and save to picked_references.txt
        picked_refs_file = os.path.join(OUTPUT_FOLDER, "picked_references.txt")
        with open(picked_refs_file, 'a', encoding='utf-8') as f:
            f.write(f"\nCitation Check Results - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Original Query: {user_query}\n\n")
            
            for url, result in zip(urls, fetch_results):
                # Get fallback metadata using normalized URL
                normalized_url = url.strip().rstrip('/').replace('\\', '/').lower()
                fallback_metadata = url_search_map.get(normalized_url, {})

                if isinstance(result, Exception):
                    root_logger.error(f"Failed to fetch {url}: {result}")
                    # Don't mark as error if we have good fallback metadata
                    if fallback_metadata and ("title" in fallback_metadata or "queries" in fallback_metadata):
                        full_text = ""
                        error_flag = False
                        metadata = fallback_metadata
                    else:
                        full_text = ""
                        error_flag = True
                        metadata = {}
                    f.write(f"URL (Failed): {url}\n")
                    f.write(f"Error: {str(result)}\n\n")
                    results["summary"]["failed_fetches"] += 1
                else:
                    full_text, error_flag, metadata = result
                    if error_flag and fallback_metadata and ("title" in fallback_metadata or "queries" in fallback_metadata):
                        error_flag = False  # Override error if we have good fallback
                        metadata = fallback_metadata
                    f.write(f"URL: {url}\n")
                    
                    if error_flag:
                        results["summary"]["failed_fetches"] += 1
                    else:
                        results["summary"]["successful_fetches"] += 1

                evaluation_text = single_citation_check(
                    call_deepseek=call_deepseek,
                    url=url,
                    full_text=full_text,
                    fallback_metadata=fallback_metadata,
                    question=user_query,
                    draft_context=content
                )

                # Consider both evaluation and metadata quality for decision
                decision = "REJECT"
                if "ACCEPT" in evaluation_text.upper() or (
                    fallback_metadata and 
                    ("queries" in fallback_metadata or "title" in fallback_metadata)
                ):
                    decision = "ACCEPT"

                # Create the citation check result
                citation_result = {
                    "url": url,
                    "evaluation": evaluation_text,
                    "has_full_text": not error_flag,
                    "source": fallback_metadata.get('queries', ['Unknown'])[0] if fallback_metadata.get('queries') else fallback_metadata.get('source_query', 'Unknown'),
                    "decision": decision,
                    "metadata": fallback_metadata if fallback_metadata else metadata
                }

                # Log to session file
                log_session_data(
                    data=citation_result,
                    filename="session_citations_check.json"
                )

                # Store in results dict as before
                results["citations"][url] = citation_result
                
                f.write(f"Evaluation:\n{evaluation_text}\n")
                f.write("-" * 80 + "\n\n")
    
    # Add basic statistics to summary
    results["summary"]["accepted_citations"] = sum(1 for r in results["citations"].values() if r["decision"] == "ACCEPT")
    results["summary"]["rejected_citations"] = sum(1 for r in results["citations"].values() if r["decision"] == "REJECT")
    
    return results




def init_selenium_driver():
    """Initialize Selenium with enhanced stealth options, error handling and automatic ChromeDriver management"""
    from selenium.webdriver.chrome.service import Service
    from webdriver_manager.chrome import ChromeDriverManager
    
    options = Options()
    options.add_argument("--headless")
    
    # Enhanced stealth options
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-gpu")
    options.add_argument("--lang=en-US,en;q=0.9")
    options.add_argument("--disable-features=IsolateOrigins,site-per-process")
    
    # Additional stealth options
    options.add_argument("--disable-web-security")
    options.add_argument("--allow-running-insecure-content")
    options.add_argument("--disable-notifications")
    options.add_argument("--disable-popup-blocking")
    
    # Set a rotating user agent
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/120.0.0.0 Safari/537.36"
    ]
    options.add_argument(f"user-agent={random.choice(user_agents)}")
    
    # Add common headers
    options.add_argument("accept=text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8")
    options.add_argument("accept-encoding=gzip, deflate, br")
    options.add_argument("accept-language=en-US,en;q=0.9")
    options.add_argument("upgrade-insecure-requests=1")
    
    # Window size
    options.add_argument("--window-size=1920,1080")
    
    # Disable automation flags
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    
    try:
        # Use webdriver_manager to automatically download and manage ChromeDriver
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        
        # Execute CDP commands to make detection harder
        driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
            "source": """
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined
                });
                Object.defineProperty(navigator, 'plugins', {
                    get: () => [1, 2, 3, 4, 5]
                });
                Object.defineProperty(navigator, 'languages', {
                    get: () => ['en-US', 'en']
                });
                window.chrome = {
                    runtime: {}
                };
                
                // Additional stealth
                const originalQuery = window.navigator.permissions.query;
                window.navigator.permissions.query = (parameters) => (
                    parameters.name === 'notifications' ?
                        Promise.resolve({state: Notification.permission}) :
                        originalQuery(parameters)
                );
            """
        })
        
        return driver
    except Exception as e:
        root_logger.error(f"Failed to initialize Selenium driver: {e}")
        raise

def process_html(html: str) -> str:
    """Convert HTML to clean Markdown text using html2text and clean noisy symbols"""
    try:
        # First try with html2text
        h = html2text.HTML2Text()
        h.ignore_links = False
        h.ignore_images = True
        h.ignore_emphasis = False
        h.body_width = 0  # Disable line wrapping
        
        try:
            processed_text = h.handle(html)
            return clean_text(processed_text)
        except Exception as e:
            root_logger.debug(f"html2text processing failed: {e}")
            
        # Fallback to BeautifulSoup with error handling
        try:
            soup = BeautifulSoup(html, 'html.parser')
            text = soup.get_text(separator=' ', strip=True)
            return clean_text(text)
        except Exception as e:
            root_logger.debug(f"BeautifulSoup processing failed: {e}")
            
        # Last resort: try to extract any readable text
        text = re.sub(r'<[^>]+>', ' ', html)
        return clean_text(text)

    except Exception as e:
        root_logger.error(f"HTML processing failed: {e}")
        return ""

def fetch_with_selenium(url: str) -> Tuple[str, bool, dict]:
    """Enhanced Selenium fallback with improved content extraction strategies"""
    print(f"\n⚠️ Attempting Selenium fallback for: {url}")
    driver = None
    try:
        driver = init_selenium_driver()
        driver.set_page_load_timeout(30)
        
        print(f"Navigating to {url}")
        driver.get(url)
        
        # Enhanced wait conditions
        wait = WebDriverWait(driver, 20)
        wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
        
        # Check for anti-bot measures
        page_source = driver.page_source.lower()
        if any(pattern in page_source for pattern in [
            "access denied", "captcha", "robot check", "please verify", 
            "security check", "blocked", "too many requests"
        ]):
            raise Exception("Access blocked by website anti-bot measures")
        
        # Scroll for dynamic content
        for _ in range(3):  # Scroll multiple times for lazy loading
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1)
        
        # Enhanced content extraction strategies
        content_strategies = [
            # Strategy 1: Main content areas with expanded selectors
            lambda: " ".join([
                elem.text for elem in driver.find_elements(By.CSS_SELECTOR, 
                "article, main, .content, .article, [role='main'], .post-content, .entry-content, #main-content")
            ]),
            # Strategy 2: Structured content
            lambda: " ".join([
                elem.text for elem in driver.find_elements(By.CSS_SELECTOR, 
                "h1, h2, h3, p, li, td, th, blockquote")
            ]),
            # Strategy 3: Paragraph text
            lambda: " ".join([
                elem.text for elem in driver.find_elements(By.TAG_NAME, "p")
            ]),
            # Strategy 4: All visible text
            lambda: driver.find_element(By.TAG_NAME, "body").text,
            # Strategy 5: HTML to text conversion
            lambda: process_html(driver.page_source)
        ]
        
        extracted_text = ""
        for i, strategy in enumerate(content_strategies, 1):
            try:
                extracted_text = strategy()
                if len(extracted_text.strip()) > 200:
                    root_logger.info(f"Content extraction succeeded with strategy {i}")
                    break
            except Exception as e:
                root_logger.debug(f"Content extraction strategy {i} failed: {e}")
                continue
        
        if not extracted_text or len(extracted_text.strip()) < 200:
            return "", True, {"error": "Insufficient content extracted"}
        
        # Clean the extracted text
        cleaned_text = clean_text(extracted_text)
        print(f"Successfully extracted {len(cleaned_text)} characters")
        return cleaned_text, False, {"source": "selenium"}
        
    except Exception as e:
        error_msg = str(e)
        root_logger.error(f"Selenium fallback failed: {error_msg}")
        return "", True, {"error": f"Selenium error: {error_msg}"}
        
    finally:
        if driver:
            try:
                driver.quit()
            except Exception as e:
                root_logger.error(f"Error closing Selenium driver: {e}")

# Disable verbose Selenium logging
selenium_logger = logging.getLogger('selenium')
selenium_logger.setLevel(logging.WARNING)

# Disable urllib3 debug logging
urllib3_logger = logging.getLogger('urllib3')
urllib3_logger.setLevel(logging.WARNING)


async def fetch_with_semantic_scholar(url: str) -> Tuple[str, bool, dict]:
    """
    Fetch paper details from Semantic Scholar with enhanced error handling.
    
    Args:
        url: Semantic Scholar paper URL
    
    Returns:
        Tuple of (text content, error flag, metadata dict)
    """
    try:
        print("\nTrying Semantic Scholar API...")
        # Extract DOI or other identifiers
        doi = None
        if 'doi.org' in url:
            doi = url.split('doi.org/')[-1]
            print(f"Found DOI: {doi}")
        elif 'sciencedirect.com' in url and 'pii' in url:
            pii = url.split('pii/')[-1].split('/')[0]
            print(f"Found PII: {pii}, searching for DOI...")
            search = sch.SemanticScholar()
            papers = search.search_paper(pii)
            if papers and len(papers) > 0 and hasattr(papers[0], 'doi'):
                doi = papers[0].doi
                print(f"Found DOI from PII: {doi}")

        if not doi:
            print("No DOI found for Semantic Scholar")
            return "", True, {}  # Return empty metadata dict

        paper = sch.SemanticScholar().get_paper(f"DOI:{doi}")
        if not paper:
            print("No paper found in Semantic Scholar")
            return "", True, {}  # Return empty metadata dict

        # Extract metadata safely using object attributes
        metadata = {
            'title': paper.title if hasattr(paper, 'title') else '',
            'abstract': paper.abstract if hasattr(paper, 'abstract') else '',
            'authors': [author.name for author in paper.authors] if hasattr(paper, 'authors') else [],
            'year': paper.year if hasattr(paper, 'year') else None,
            'doi': doi,
            'venue': paper.venue if hasattr(paper, 'venue') else '',
            'citations': len(paper.citations) if hasattr(paper, 'citations') else 0,
            'references': len(paper.references) if hasattr(paper, 'references') else 0
        }

        # Combine available text fields
        text_parts = []
        if metadata['title']:
            text_parts.append(f"Title: {metadata['title']}\n")
        if metadata['abstract']:
            text_parts.append(f"Abstract: {metadata['abstract']}\n")
        if hasattr(paper, 'tldr') and paper.tldr and hasattr(paper.tldr, 'text'):
            text_parts.append(f"Summary: {paper.tldr.text}\n")

        full_text = "\n".join(text_parts)
        success = len(full_text) > 200
        print(f"Semantic Scholar {'succeeded' if success else 'failed'} to get sufficient content")
        return full_text, not success, metadata

    except AttributeError as e:
        root_logger.error(f"Semantic Scholar fetch failed: {str(e)}")
        print(f"Semantic Scholar error: {e}")
        return "", True, {}
    except Exception as e:
        root_logger.error(f"Semantic Scholar fetch failed: {str(e)}")
        print(f"Semantic Scholar error: {e}")
        return "", True, {}

def get_stealth_headers():
    """Generate headers that mimic regular browser requests"""
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/120.0.0.0 Safari/537.36"
    ]
    
    headers = {
        'User-Agent': random.choice(user_agents),
        'Accept': 'text/html,application/xhtml+xml,application/xml,application/pdf;q=0.9,image/webp,image/apng,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.9',
        'Accept-Encoding': 'gzip, deflate, br',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Cache-Control': 'max-age=0',
        'Sec-Fetch-Dest': 'document',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-Site': 'none',
        'Sec-Fetch-User': '?1',
        'Sec-Ch-Ua': '"Not A(Brand";v="99", "Google Chrome";v="120", "Chromium";v="120"',
        'Sec-Ch-Ua-Mobile': '?0',
        'Sec-Ch-Ua-Platform': '"Windows"',
        'DNT': '1',  # Do Not Track
    }
    
    return headers

async def fetch_with_crossref(url: str) -> Tuple[str, bool, dict]:
    """Attempt to fetch content using Crossref API"""
    try:
        print("\nTrying Crossref API...")
        cr = habanero.Crossref(mailto="your.email@ntnu.no")
        
        # Extract DOI
        doi = None
        if 'doi.org' in url:
            doi = url.split('doi.org/')[-1]
            print(f"Found DOI: {doi}")
        elif 'sciencedirect.com' in url and 'pii' in url:
            pii = url.split('pii/')[-1].split('/')[0]
            print(f"Found PII: {pii}, searching for DOI...")
            results = cr.works(query=pii, limit=1)
            if results['message']['items']:
                doi = results['message']['items'][0].get('DOI')
                print(f"Found DOI from PII: {doi}")

        if not doi:
            print("No DOI found for Crossref")
            return "", True, {}  # Return empty metadata dict

        work = cr.works(ids=doi)
        if not work or 'message' not in work:
            print("No work found in Crossref")
            return "", True, {}  # Return empty metadata dict

        work_data = work['message']
        text_parts = []
        metadata = {
            'doi': doi,
            'title': work_data.get('title', [None])[0],
            'year': work_data.get('published-print', {}).get('date-parts', [[None]])[0][0],
            'journal': work_data.get('container-title', [None])[0],
            'authors': []
        }
        
        if work_data.get('author'):
            metadata['authors'] = [f"{a.get('given', '')} {a.get('family', '')}" for a in work_data['author']]
            
        # Build text content
        if metadata['title']:
            text_parts.append(f"Title: {metadata['title']}\n")
        if work_data.get('abstract'):
            text_parts.append(f"Abstract: {work_data['abstract']}\n")
        if metadata['authors']:
            text_parts.append(f"Authors: {', '.join(metadata['authors'])}\n")
        if metadata['year']:
            text_parts.append(f"Published: {metadata['year']}\n")
        if metadata['journal']:
            text_parts.append(f"Journal: {metadata['journal']}\n")

        full_text = "\n".join(text_parts)
        success = len(full_text) > 200
        print(f"Crossref {'succeeded' if success else 'failed'} to get sufficient content")
        return full_text, not success, metadata

    except Exception as e:
        root_logger.error(f"Crossref fetch failed: {e}")
        print(f"Crossref error: {e}")
        return "", True, {}  # Return empty metadata dict

def fetch_with_scrapingbee(url: str) -> Tuple[str, bool, dict]:
    """
    Enhanced ScrapingBee client with:
    - Retry logic with exponential backoff
    - Improved content validation
    - Better encoding detection
    - Detailed error logging
    Returns (content, error_flag, metadata) tuple.
    """
    try:
        client = ScrapingBeeClient(api_key=SCRAPINGBEE_API_KEY)
        headers = get_stealth_headers()
        
        for attempt in range(3):
            try:
                response = client.get(
                    url,
                    params={
                        'premium_proxy': True,
                        'country_code': 'us',
                        'render_js': True,
                        'wait_for': 5000,
                        'timeout': 30000,
                        'block_resources': "image,font,stylesheet"
                    },
                    headers=headers
                )
                
                if response.status_code == 200:
                    detector = chardet.UniversalDetector()
                    detector.feed(response.content)
                    detector.close()
                    encoding = detector.result['encoding'] or 'utf-8'
                    
                    try:
                        content = response.content.decode(encoding, errors='replace')
                    except UnicodeDecodeError:
                        content = response.content.decode('utf-8', errors='replace')
                    
                    cleaned = process_html(content)
                    if 5000 > len(cleaned) > 100:
                        root_logger.info(f"ScrapingBee success on attempt {attempt+1}")
                        return cleaned, False, {}
                    
                    root_logger.warning(f"Insufficient content length: {len(cleaned)}")
                    
                elif response.status_code in [500, 503, 504]:
                    root_logger.warning(f"ScrapingBee server error, retrying... (attempt {attempt+1}/3)")
                    time.sleep(2 ** attempt)
                    continue
                    
            except Exception as e:
                root_logger.error(f"ScrapingBee attempt {attempt+1} failed: {str(e)}")
                if attempt < 2:
                    time.sleep(5)
        
        return "", True, {}
        
    except Exception as e:
        root_logger.error(f"ScrapingBee final failure: {str(e)}")
        return "", True, {}



def fetch_with_parsehub(url: str) -> Tuple[str, bool, dict]:
    """
    Fetch URL content using ParseHub.
    Returns (content, error_flag, metadata) tuple.
    """
    try:
        # Start a new run
        start_url = f"https://www.parsehub.com/api/v2/projects/{PARSEHUB_PROJECT_TOKEN}/run"
        params = {
            "api_key": PARSEHUB_API_KEY,
            "start_url": url,
            "send_email": "0"
        }
        response = requests.post(start_url, data=params)
        if response.status_code != 200:
            return "", True, {}
            
        run_token = response.json().get('run_token')
        if not run_token:
            return "", True, {}
            
        # Wait for the run to complete (with timeout)
        max_wait = 60  # Maximum wait time in seconds
        start_time = time.time()
        while True:
            if time.time() - start_time > max_wait:
                return "", True, {}
                
            get_data_url = f"https://www.parsehub.com/api/v2/runs/{run_token}/data"
            params = {"api_key": PARSEHUB_API_KEY}
            response = requests.get(get_data_url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'complete':
                    content = json.dumps(data.get('data', {}))
                    # ParseHub returns structured JSON, so we don't process it as HTML
                    if len(content) > 100:
                        return content, False, {"source": "parsehub"}
                    return "", True, {}
                    
            time.sleep(5)  # Wait before checking again
            
    except Exception as e:
        root_logger.error(f"ParseHub fetch failed: {str(e)}")
        return "", True, {}



def extract_metadata(text: str) -> dict:
    """Extract citation metadata from text content"""
    # DOI detection
    doi_match = re.search(r'\b(10\.\d{4,9}/[-._;()/:A-Z0-9]+)\b', text, re.I)
    # Year detection
    year_match = re.search(r'(19|20)\d{2}', text)
    # Author detection (simple pattern)
    author_match = re.search(r'by\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})', text)
    
    return {
        'doi': doi_match.group(1) if doi_match else None,
        'year': year_match.group(0) if year_match else None,
        'authors': [author_match.group(1)] if author_match else []
    }

async def enrich_with_crossref(doi: str) -> dict:
    """Enrich metadata using Crossref API"""
    try:
        cr = habanero.Crossref()
        work = cr.works(ids=doi)
        if work and 'message' in work:
            msg = work['message']
            return {
                'authors': [f"{a['given']} {a['family']}" for a in msg.get('author', [])],
                'year': msg.get('issued', {}).get('date-parts', [[None]])[0][0],
                'journal': msg.get('container-title', [None])[0],
                'title': msg.get('title', [None])[0]
            }
    except Exception as e:
        print(f"Crossref enrichment failed: {str(e)}")
    return {}


# Make sure to install Playwright and set it up:
# pip install playwright
# playwright install chromium

from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout
from bs4 import BeautifulSoup
import re

async def fetch_with_playwright(url: str) -> tuple:
    """
    Attempt to fetch and extract content from a URL using Playwright.
    
    Returns:
        A tuple (text, error_flag, metadata), where:
         - text: the extracted text (or an empty string if extraction fails)
         - error_flag: a boolean indicating whether an error occurred (True if error)
         - metadata: an (optional) dictionary of metadata (empty in this basic example)
    """
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True,
                args=[
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-gpu",
                    "--disable-web-security"
                ]
            )
            
            context = await browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                viewport={"width": 1920, "height": 1080},
                ignore_https_errors=True
            )
            
            # Add stealth script
            await context.add_init_script("""
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined
                });
                Object.defineProperty(navigator, 'plugins', {
                    get: () => [1, 2, 3, 4, 5]
                });
                Object.defineProperty(navigator, 'languages', {
                    get: () => ['en-US', 'en']
                });
            """)
            
            page = await context.new_page()
            
            try:
                response = await page.goto(
                    url, 
                    wait_until="networkidle", 
                    timeout=45000
                )
                
                if not response:
                    root_logger.error(f"No response received for {url}")
                    return "", True, {}
                    
                # Wait for content to load
                await page.wait_for_load_state("domcontentloaded")
                await page.wait_for_selector("body", timeout=5000)
                
                # Enhanced content extraction with multiple strategies
                content = await page.evaluate("""() => {
                    function getContent() {
                        // Try multiple content selectors
                        const selectors = [
                            'article',
                            'main',
                            '[role="main"]',
                            '#content',
                            '.content',
                            '.article-content',
                            '.post-content'
                        ];
                        
                        for (const selector of selectors) {
                            const element = document.querySelector(selector);
                            if (element && element.innerText.trim().length > 200) {
                                return element.innerText;
                            }
                        }
                        
                        // Fallback to paragraphs
                        const paragraphs = Array.from(document.querySelectorAll('p'));
                        return paragraphs.map(p => p.innerText).join("\\n\\n");
                    }
                    return getContent();
                }""")
                
                await browser.close()
                
                if content and len(content.strip()) > 100:
                    cleaned = re.sub(r'\s+', ' ', content).strip()
                    return cleaned, False, {}
                    
                root_logger.warning(f"Insufficient content extracted from {url}")
                return "", True, {}
                
            except PlaywrightTimeout:
                root_logger.error(f"Timeout accessing {url}")
                return "", True, {}
            except Exception as e:
                root_logger.error(f"Navigation error for {url}: {e}")
                return "", True, {}
                
    except Exception as e:
        root_logger.error(f"Playwright error: {str(e)}\nCall log:\n{traceback.format_exc()}")
        return "", True, {}



async def fetch_with_google_scholar(doi: str) -> Tuple[str, bool, dict]:
    """
    Attempt to fetch paper content using Google Scholar search with DOI.
    Looks for both PDF and HTML full-text links.
    Returns (content, error_flag, metadata) tuple.
    """
    try:
        print("\nTrying Google Scholar...")
        search_url = f"https://scholar.google.com/scholar?q={doi}"
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True,
                args=['--no-sandbox', '--disable-dev-shm-usage']
            )
            context = await browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                viewport={"width": 1920, "height": 1080}
            )
            
            # Add stealth script
            await context.add_init_script("""
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined
                });
            """)
            
            page = await context.new_page()
            
            try:
                await page.goto(search_url, wait_until="networkidle", timeout=30000)
                
                # Look for both PDF and full-text links
                links = await page.evaluate("""() => {
                    const allLinks = Array.from(document.querySelectorAll('a'));
                    const results = {
                        pdfLinks: [],
                        fullTextLinks: []
                    };
                    
                    for (const link of allLinks) {
                        const href = link.href.toLowerCase();
                        const text = link.textContent.toLowerCase();
                        
                        // Check for PDF links
                        if (href.includes('.pdf') || text.includes('pdf')) {
                            results.pdfLinks.push(link.href);
                        }
                        // Check for full-text links
                        else if (
                            text.includes('full text') || 
                            text.includes('fulltext') ||
                            text.includes('free') ||
                            text.includes('html') ||
                            href.includes('doi.org') ||
                            (link.closest('.gs_or_ggsm') && !text.includes('cite'))
                        ) {
                            results.fullTextLinks.push(link.href);
                        }
                    }
                    return results;
                }""")
                
                print(f"Found {len(links['pdfLinks'])} PDF links and {len(links['fullTextLinks'])} full-text links")
                
                # Try PDF links first
                for pdf_url in links['pdfLinks']:
                    print(f"Trying PDF link: {pdf_url}")
                    try:
                        async with aiohttp.ClientSession() as session:
                            async with session.get(pdf_url, headers=get_stealth_headers()) as response:
                                if response.status == 200:
                                    content = await response.read()
                                    pdf_file = BytesIO(content)
                                    
                                    loop = asyncio.get_event_loop()
                                    text = await loop.run_in_executor(None, process_pdf, pdf_file)
                                    
                                    if text and len(text.strip()) > 200:
                                        print("Successfully extracted text from Google Scholar PDF")
                                        return text, False, extract_metadata(text)
                    except Exception as e:
                        print(f"Failed to fetch PDF from {pdf_url}: {str(e)}")
                        continue
                
                # Try full-text links
                for full_text_url in links['fullTextLinks']:
                    print(f"Trying full-text link: {full_text_url}")
                    try:
                        # Use the main fetch function to handle the full-text URL
                        text, error, metadata = await fetch_full_text_async(
                            full_text_url, 
                            session=aiohttp.ClientSession(),  # Create new session
                            semaphore=asyncio.Semaphore(5)
                        )
                        
                        if not error and text:
                            print("Successfully extracted text from full-text link")
                            return text, False, metadata
                    except Exception as e:
                        print(f"Failed to fetch full text from {full_text_url}: {str(e)}")
                        continue
                
                print("No accessible content found on Google Scholar")
                return "", True, {}
                
            except Exception as e:
                print(f"Error accessing Google Scholar: {str(e)}")
                return "", True, {}
            
            finally:
                await browser.close()
                
    except Exception as e:
        root_logger.error(f"Google Scholar fetch failed: {str(e)}")
        return "", True, {}

def extract_doi_from_url(url: str) -> Optional[str]:
    """
    Extract DOI from URL using various common patterns.
    Returns None if no DOI is found.
    """
    # Common DOI patterns
    doi_patterns = [
        r'doi/(?:abs/|full/)?(\d{2}\.\d{4,}/[-._;()/:\w]+)',  # matches doi/10.1234/etc
        r'doi\.org/(\d{2}\.\d{4,}/[-._;()/:\w]+)',           # matches doi.org/10.1234/etc
        r'dx\.doi\.org/(\d{2}\.\d{4,}/[-._;()/:\w]+)',       # matches dx.doi.org/10.1234/etc
        r'/(\d{2}\.\d{4,}/[-._;()/:\w]+)'                    # matches /10.1234/etc anywhere
    ]
    
    for pattern in doi_patterns:
        match = re.search(pattern, url)
        if match:
            doi = match.group(1)
            # Validate DOI format
            if doi.startswith('10.'):
                return doi
    return None



async def fetch_full_text_async(url: str, session: aiohttp.ClientSession, semaphore: asyncio.Semaphore) -> Tuple[str, bool, dict]:
    """Enhanced fetch content with improved error handling and metadata extraction"""
    # Comprehensive URL validation and cleanup
    if not url or not isinstance(url, str):
        root_logger.error(f"Invalid or missing URL provided: {url}")
        return "", True, {"error": "Invalid or missing URL format"}

    # Clean up URL if it contains prefixes
    if url.startswith('Missing reference for URL:'):
        url = url.replace('Missing reference for URL:', '').strip()
    elif 'URL:' in url:
        url = url.split('URL:')[-1].strip()

    # Validate URL structure
    try:
        result = urlparse(url)
        if not all([result.scheme, result.netloc]):
            root_logger.error(f"Malformed URL structure: {url}")
            return "", True, {"error": "Malformed URL structure"}
    except Exception as e:
        root_logger.error(f"URL parsing error: {str(e)}")
        return "", True, {"error": f"URL parsing error: {str(e)}"}

    headers = get_stealth_headers()
    root_logger.debug(f"Starting fetch for URL: {url}")
    metadata = {}

    # Try Playwright first
    print("\nTrying Playwright extraction first...")
    text, error, playwright_metadata = await fetch_with_playwright(url)
    if not error and text:
        print("Playwright extraction succeeded")
        metadata.update(playwright_metadata)
        return text, False, metadata
    print("Playwright attempt failed, trying other methods...")
    
    # Check for DOI in URL
    doi = extract_doi_from_url(url)
    if doi:
        print(f"Found DOI in URL: {doi}")
        # Try Google Scholar first for DOIs
        text, error, gs_metadata = await fetch_with_google_scholar(doi)
        if not error and text:
            metadata.update(gs_metadata)
            return text, False, metadata
        print("Google Scholar attempt failed, trying other methods...")
        
        # Try Semantic Scholar and Crossref as backups
        for method in [fetch_with_semantic_scholar, fetch_with_crossref]:
            text, error, method_metadata = await method(url)
            if not error and text:
                metadata.update(method_metadata)
                return text, False, metadata
    
    # Try scraping services
    scraping_services = [
        ('ScrapingBee', SCRAPINGBEE_API_KEY, fetch_with_scrapingbee),
        ('ParseHub', PARSEHUB_API_KEY, fetch_with_parsehub)
    ]
    
    for service_name, api_key, fetch_func in scraping_services:
        if api_key and api_key != f"YOUR_{service_name.upper()}_API_KEY":
            print(f"\nTrying {service_name} API...")
            loop = asyncio.get_event_loop()
            text, error, service_metadata = await loop.run_in_executor(None, fetch_func, url)
            if not error:
                root_logger.info(f"Successfully fetched via {service_name}")
                metadata.update(service_metadata)
                return text, False, metadata
            print(f"{service_name} attempt failed, trying next method...")

    # Handle DOI URLs
    if 'doi.org' in url:
        try:
            doi = url.split('doi.org/')[-1]
            resolved_url = f"https://dx.doi.org/{doi}"
            print(f"Resolving DOI: {doi}")
            
            # Try academic APIs first
            for method in [fetch_with_google_scholar, fetch_with_semantic_scholar, fetch_with_crossref]:
                text, error, method_metadata = await method(doi)
                if not error and text:
                    metadata.update(method_metadata)
                    return text, False, metadata
            
            # Try resolving DOI URL
            async with semaphore:
                async with session.get(resolved_url, headers=headers, allow_redirects=True, timeout=30) as response:
                    if response.status == 200:
                        url = str(response.url)
                        print(f"DOI resolved to: {url}")
                    else:
                        print(f"Failed to resolve DOI {doi}")
                        return "", True, {"error": "DOI resolution failed"}
        except Exception as e:
            print(f"Error resolving DOI: {e}")
            return "", True, {"error": f"DOI error: {str(e)}"}

    # Standard content fetch with retries
    for attempt in range(3):
        try:
            async with semaphore:
                async with session.get(url, headers=headers, timeout=REQUEST_TIMEOUT) as response:
                    print(f"Attempt {attempt + 1}: Status {response.status}")
                    
                    if response.status != 200:
                        if attempt < 2:
                            print(f"Retrying after failed attempt...")
                            await asyncio.sleep(1)
                            continue
                        
                        # Try fallback methods
                        print("Trying synchronous requests...")
                        try:
                            loop = asyncio.get_event_loop()
                            response = await loop.run_in_executor(
                                None, 
                                lambda: requests.get(
                                    url, 
                                    headers=get_stealth_headers(),
                                    timeout=REQUEST_TIMEOUT,
                                    verify=True,
                                    allow_redirects=True
                                )
                            )
                            if response.status_code == 200:
                                text = process_html(response.text)
                                if text and len(text.strip()) > 100:
                                    extracted_metadata = await extract_metadata_from_content(text)
                                    return clean_text(text), False, extracted_metadata
                        except Exception as re:
                            print(f"Synchronous requests failed: {str(re)}")
                        
                        # Try Selenium as last resort
                        print("Trying Selenium fallback...")
                        loop = asyncio.get_event_loop()
                        selenium_text, selenium_error, selenium_metadata = await loop.run_in_executor(
                            None, fetch_with_selenium, url
                        )
                        return selenium_text, selenium_error, selenium_metadata

                    # Process content based on type
                    content_type = response.headers.get('Content-Type', '').lower()
                    print(f"Content-Type: {content_type}")
                    content = await response.read()
                    
                    # Handle PDFs
                    if 'application/pdf' in content_type or url.lower().endswith('.pdf'):
                        return await handle_pdf_content(content, url)
                    
                    # Handle HTML
                    return await handle_html_content(content, url)

        except Exception as e:
            print(f"Error on attempt {attempt + 1}: {str(e)}")
            if attempt < 2:
                await asyncio.sleep(1)
                continue
            
            # Final fallback to Selenium
            print("All attempts failed, trying Selenium fallback...")
            try:
                loop = asyncio.get_event_loop()
                selenium_text, selenium_error, selenium_metadata = await loop.run_in_executor(
                    None, fetch_with_selenium, url
                )
                return selenium_text, selenium_error, selenium_metadata
            except Exception as se:
                root_logger.error(f"Selenium fallback failed: {str(se)}")
                return "", True, {"error": f"All fetch attempts failed: {str(se)}"}

    return "", True, {"error": "Maximum fetch attempts reached"}

async def handle_pdf_content(content: bytes, url: str) -> Tuple[str, bool, dict]:
    """Handle PDF content extraction with metadata"""
    print("Processing PDF content...")
    pdf_file = BytesIO(content)
    loop = asyncio.get_event_loop()
    text = await loop.run_in_executor(None, process_pdf, pdf_file)
    
    if not text:
        print("No text extracted from PDF")
        return "", True, {"error": "PDF extraction failed"}
    
    print(f"Successfully extracted {len(text)} characters from PDF")
    metadata = await extract_metadata_from_content(text)
    metadata.update({"content_type": "pdf", "source_url": url})
    return text, False, metadata

async def handle_html_content(content: bytes, url: str) -> Tuple[str, bool, dict]:
    """Handle HTML content extraction with metadata"""
    print("Processing HTML content...")
    try:
        text = content.decode('utf-8')
        soup = BeautifulSoup(text, 'html.parser')
        
        # Remove unwanted elements
        for element in soup.find_all(['script', 'style', 'nav', 'header', 'footer']):
            element.decompose()
        
        # Extract content using multiple strategies
        content_elements = (
            soup.find_all('p') or 
            soup.find_all(['article', 'main']) or 
            soup.find_all('div', class_=lambda x: x and any(term in str(x).lower() 
                for term in ['content', 'article', 'main', 'body']))
        )
        
        text = clean_text(' '.join(el.get_text(strip=True) for el in content_elements))
        print(f"Extracted {len(text)} characters of text")
        
        if len(text) < 100:
            print("Warning: Minimal content extracted")
            return "", True, {"error": "Insufficient content length"}
        
        metadata = await extract_metadata_from_content(text)
        metadata.update({
            "content_type": "html",
            "source_url": url,
            "title": soup.title.string if soup.title else None
        })
        
        return text, False, metadata
        
    except Exception as e:
        print(f"HTML processing failed: {str(e)}")
        return "", True, {"error": f"HTML processing failed: {str(e)}"}

async def extract_metadata_from_content(text: str) -> dict:
    """Extract metadata from content using multiple methods"""
    metadata = {}
    
    # Try multiple extraction methods
    try:
        # Basic metadata
        metadata.update(extract_metadata(text))
        
        # Additional metadata extraction methods can be added here
        
    except Exception as e:
        root_logger.error(f"Metadata extraction failed: {str(e)}")
    
    return metadata

def store_search_results(search_results, filename=None):
    """Store search results with detailed logging"""
    print("\n=== STORING SEARCH RESULTS ===")
    try:
        if not filename:
            filename = os.path.join(OUTPUT_FOLDER, "search_results", "session_search_results.json")
        print(f"Writing to: {filename}")
        
        # Store raw results alongside the processed results
        raw_results_file = os.path.splitext(filename)[0] + "_raw.json"
        with open(raw_results_file, 'w', encoding='utf-8') as f:
            json.dump(search_results, f, indent=2, ensure_ascii=False)
        print(f"Stored raw search results in: {raw_results_file}")
        
        # Update session storage only
        update_search_results(search_results)
        
        # Convert and validate data before saving
        url_map = {}
        invalid_count = 0
        for item in search_results:
            if not item or 'results' not in item:
                root_logger.warning(f"Skipping invalid result item: {item}")
                invalid_count += 1
                continue
                
            results = item['results']
            if isinstance(results, str):
                try:
                    results = json.loads(results)
                    print("Successfully parsed JSON string results")
                except json.JSONDecodeError as e:
                    root_logger.error(f"Failed to parse JSON: {e}")
                    invalid_count += 1
                    continue
            
            if isinstance(results, dict) and 'results' in results:
                for entry in results['results']:
                    if isinstance(entry, dict):
                        url = entry.get('url')
                        if url:
                            url_map[url] = entry
                            url_map[url]['source_query'] = item.get('query', '')
                            url_map[url]['original_url'] = url
                            # Add detailed logging of each result
                            log_entry = {
                                'url': url,
                                'title': entry.get('title', 'No title'),
                                'snippet': entry.get('snippet', 'No snippet'),
                                'source_query': item.get('query', '')
                            }
                            root_logger.debug(f"Search result found: {json.dumps(log_entry, indent=2)}")

        # Add metadata to the saved results
        output_data = {
            "metadata": {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "session_id": os.path.basename(OUTPUT_FOLDER),
                "total_results": len(url_map),
                "invalid_items": invalid_count,
                "search_queries": [item.get('query', '') for item in search_results if item]
            },
            "url_map": url_map
        }

        # Save to JSON file
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        # Save readable version
        txt_filename = os.path.splitext(filename)[0] + "_readable.txt"
        with open(txt_filename, 'w', encoding='utf-8') as f:
            f.write(f"Search Results Summary\n")
            f.write(f"Session: {output_data['metadata']['session_id']}\n")
            f.write(f"Generated on: {output_data['metadata']['timestamp']}\n")
            f.write(f"Total References: {len(url_map)}\n")
            f.write(f"Invalid Items: {invalid_count}\n\n")
            f.write("Search Queries Used:\n")
            for query in output_data['metadata']['search_queries']:
                if query:
                    f.write(f"- {query}\n")
            f.write("\nReferences Found:\n\n")
            
            for url, entry in url_map.items():
                f.write(f"URL: {url}\n")
                f.write(f"Title: {entry.get('title', 'No title')}\n")
                f.write(f"Query: {entry.get('source_query', 'Unknown query')}\n")
                if 'snippet' in entry:
                    f.write(f"Snippet: {entry['snippet'][:2000]}...\n")
                f.write("\n")
        
        print(f"Successfully saved to {filename}")
        print(f"Also saved readable version to {txt_filename}")
        return filename
        
    except Exception as e:
        root_logger.error(f"ERROR storing results: {str(e)}", exc_info=True)
        print(f"ERROR storing results: {str(e)}")
        print("Full error details:", exc_info=True)
        return None

def create_url_search_map(search_results: list) -> Dict:
    """Create mapping of URLs to their search result data with metadata merging."""
    url_search_map = {}
    for search_item in search_results:
        if not isinstance(search_item, dict) or 'results' not in search_item:
            continue

        results = search_item['results']
        if isinstance(results, str):
            try:
                results = json.loads(results)
            except json.JSONDecodeError:
                print("Failed to parse JSON in search results")
                continue
        
        if not isinstance(results, dict) or 'results' not in results:
            continue
        
        for entry in results.get('results', []):
            if isinstance(entry, dict):
                url = entry.get('url')
                if not url:
                    continue
                # Normalize URL for comparison
                normalized_url = url.strip().rstrip('/').replace('\\', '/').lower()
                if normalized_url in url_search_map:
                    existing_entry = url_search_map[normalized_url]
                    # Merge snippet: use the longer snippet
                    existing_snippet = existing_entry.get('snippet', '')
                    new_snippet = entry.get('snippet', '')
                    if len(new_snippet) > len(existing_snippet):
                        existing_entry['snippet'] = new_snippet
                    # Merge metadata: update fields that are missing or longer
                    for key, value in entry.items():
                        if key not in existing_entry or (isinstance(value, str) and len(value) > len(existing_entry.get(key, ''))):
                            existing_entry[key] = value
                    # Merge the source query: add if not already present
                    existing_queries = existing_entry.get('queries', [])
                    new_query = search_item.get('query', '')
                    if new_query and new_query not in existing_queries:
                        existing_queries.append(new_query)
                        existing_entry['queries'] = existing_queries
                else:
                    # Create a new entry with a queries list
                    new_entry = dict(entry)
                    new_entry['queries'] = [search_item.get('query', '')]
                    new_entry['processed_at'] = datetime.now().isoformat()
                    url_search_map[normalized_url] = new_entry
                    
    return url_search_map

def extract_json(text: str):
    """
    Attempts to extract valid JSON objects from a string.

    This updated version handles:
    1. Direct JSON parsing
    2. Multiple JSON objects embedded in text
    3. JSON within markdown code blocks
    4. Partial or malformed JSON gracefully

    Returns a list of successfully parsed JSON objects.
    If no valid JSON is found, returns an empty list.
    """


    # If text contains markdown code blocks, clean it first
    if "```" in text:
        # Extract content from code blocks
        blocks = text.split("```")
        for block in blocks:
            if block.strip().startswith("json"):
                # Remove the "json" prefix and try parsing
                text = block.replace("json", "", 1).strip()
                break

    # Try direct parse first
    try:
        parsed = json.loads(text.strip())
        return [parsed] if isinstance(parsed, dict) or isinstance(parsed, list) else []
    except json.JSONDecodeError:
        pass

    # Fallback: Find all possible JSON objects via regex
    possible_json_strings = re.findall(r'(\{.*?\}|\[.*?\])', text, flags=re.DOTALL)

    parsed_objects = []
    for candidate in possible_json_strings:
        candidate = candidate.strip()
        try:
            parsed_candidate = json.loads(candidate)
            parsed_objects.append(parsed_candidate)
        except json.JSONDecodeError:
            continue

    return parsed_objects

async def iterative_search_engine(
    user_query: str,
    manager_plan: str,
    max_search_rounds: int = MAX_SEARCH_ROUNDS,
    max_references: int = MAX_REFERENCES
) -> Tuple[List[dict], SearchHistory]:  # Changed return type
    """
    Execute multiple consecutive searches with comprehensive history tracking.
    """
    all_picked_references = []
    empty_result_count = 0
    search_history = SearchHistory()
    seen_urls = set()  # Track unique normalized URLs

    # Generate initial query
    root_logger.info("\nGenerating initial search query...")
    first_query_response = researcher_agent(
        context="",
        question=user_query,
        manager_plan=manager_plan
    )

    try:
        query_data = json.loads(first_query_response)
        current_query = query_data.get('query', first_query_response)
    except json.JSONDecodeError:
        current_query = first_query_response.strip().replace('```json', '').replace('```', '').strip()

    round_num = 1
    
    while round_num <= max_search_rounds and len(all_picked_references) < max_references:
        root_logger.info(f"\n[Search Round {round_num}] Query: {current_query}")
        
        # Skip if we've already tried this exact query
        if current_query.lower() in search_history.used_queries:
            root_logger.info("Skipping duplicate query")
            # Increment round number and try to get a new query
            round_num += 1
            try:
                next_query_response = consecutive_researcher_agent(
                    manager_plan=manager_plan,
                    user_question=user_query,
                    picked_results_so_far=all_picked_references,
                    search_history=search_history
                )

                try:
                    query_data = json.loads(next_query_response)
                    current_query = query_data.get('query', next_query_response)
                except json.JSONDecodeError:
                    current_query = next_query_response.strip().replace('```json', '').replace('```', '').strip()
                
                root_logger.info(f"Next query generated: {current_query}")
            except Exception as e:
                root_logger.error(f"Failed to generate next query after duplicate: {e}")
                break
            continue

        # Execute search with retry logic
        raw_result = {}
        max_retries = 3
        for attempt in range(max_retries):
            try:
                result_data = search_tool._run(current_query)
                raw_result = pydantic_to_dict(result_data)
                if isinstance(raw_result, str):
                    try:
                        raw_result = json.loads(raw_result)
                    except json.JSONDecodeError:
                        root_logger.error("Failed to parse string result as JSON")
                        raw_result = {}
                break
            except Exception as e:
                root_logger.error(f"Search attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    root_logger.error("All search attempts failed")
                    break
                await asyncio.sleep(2)

        # Pick relevant results
        picked_results = search_result_picker_agent(user_query, raw_result)
        # Update session search results
        if raw_result:
            update_search_results([{'query': current_query, 'results': raw_result}])
        # Get the FULL result objects from raw_result that match our picked URLs
        relevant_list = []
        if picked_results["status"] != "NO_RELEVANT_SEARCH_RESULTS":
            picked_urls = set(result["url"] for result in picked_results.get("relevant_results", []))
            # Extract full result objects from raw_result
            if "results" in raw_result:
                results = raw_result["results"]
                if isinstance(results, dict) and "results" in results:
                    results = results["results"]
                for result in results:
                    if result.get("url") in picked_urls:
                        relevant_list.append(result)  # Keep the FULL result object
        
        # Record this search attempt in history
        search_history.add_search(
            query=current_query,
            results=relevant_list,
            was_successful=picked_results["status"] != "NO_RELEVANT_SEARCH_RESULTS",
            raw_result=raw_result  # Pass the complete raw result
        )
        
        if picked_results["status"] == "NO_RELEVANT_SEARCH_RESULTS":
            empty_result_count += 1
            root_logger.warning(f"No relevant results found (count: {empty_result_count})")
        else:
            empty_result_count = 0
            
            # Enhanced duplicate handling with metadata merging
            new_relevant_list = []
            existing_refs = {
                ref.get("url", "").strip().rstrip('/').replace('\\', '/').lower(): ref 
                for ref in all_picked_references
            }
            
            for item in relevant_list:
                normalized_url = item.get("url", "").strip().rstrip('/').replace('\\', '/').lower()
                if not normalized_url:
                    continue
                    
                if normalized_url in existing_refs:
                    # Merge with existing reference
                    existing = existing_refs[normalized_url]
                    # Update snippet if new one is longer
                    if len(item.get('snippet', '')) > len(existing.get('snippet', '')):
                        existing['snippet'] = item.get('snippet', '')
                    # Merge source queries
                    existing.setdefault('queries', []).append(current_query)
                    # Merge other metadata fields
                    for key, value in item.items():
                        if key not in existing or (isinstance(value, str) and len(value) > len(existing.get(key, ''))):
                            existing[key] = value
                else:
                    # Add new reference
                    seen_urls.add(normalized_url)
                    item["source_query"] = current_query
                    item['queries'] = [current_query]
                    new_item = dict(item)
                    new_relevant_list.append(new_item)
                    existing_refs[normalized_url] = new_item
            
            if new_relevant_list:  # Only extend if we have new unique results
                all_picked_references.extend(new_relevant_list)
                root_logger.info(f"Found {len(new_relevant_list)} new unique results")
                # Log new references to session
                for result in new_relevant_list:
                    log_session_data(
                        data=result,
                        filename="session_picked_references.json"  # Remove extra search_results path
                    )
            else:
                root_logger.info("No new unique results found")

        root_logger.info(f"Total unique references collected: {len(all_picked_references)}/{max_references}")
                
        if len(all_picked_references) >= max_references:
            root_logger.info(f"\nReached {max_references} unique references, stopping iterative search.")
            break

        # Get next query using consecutive_researcher_agent with search history
        try:
            next_query_response = consecutive_researcher_agent(
                manager_plan=manager_plan,
                user_question=user_query,
                picked_results_so_far=all_picked_references,
                search_history=search_history
            )

            try:
                query_data = json.loads(next_query_response)
                current_query = query_data.get('query', next_query_response)
            except json.JSONDecodeError:
                current_query = next_query_response.strip().replace('```json', '').replace('```', '').strip()
                
            root_logger.info(f"Next query generated: {current_query}")
            
        except Exception as e:
            root_logger.error(f"Failed to generate next query: {e}")
            break

        round_num += 1

    # Final summary
    root_logger.info(f"\nSearch completed: {len(all_picked_references)} unique references found in {round_num} rounds")
    return all_picked_references, search_history  # Return both


def integrate_drafts(draft1, draft2, draft3, question):
    """
    Integrates multiple drafts into a single improved version with model fallbacks.
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a final integrator combining multiple drafts into one superior version. Follow these rules:\n"
                "1. Merge the strongest elements from all drafts\n"
                "2. Maintain academic rigor and citation accuracy\n"
                "3. Preserve all valuable citations and references\n"
                "4. Ensure logical flow and coherent structure\n"
                "5. Keep the most insightful analysis from each draft\n"
                "6. Maintain consistent academic tone and style\n"
                "7. Integrate complementary perspectives thoughtfully\n"
                "8. Preserve technical accuracy and precision\n"
                "9. Keep all relevant examples and evidence\n"
                "10. Ensure comprehensive coverage of the topic"
            )
        },
        {
            "role": "user",
            "content": (
                f"### RESEARCH QUESTION ###\n{question}\n\n"
                f"### DRAFT 1 (GEMINI) ###\n{draft1}\n\n"
                f"### DRAFT 2 (DEEPSEEK) ###\n{draft2}\n\n"
                f"### DRAFT 3 (O3-MINI) ###\n{draft3}\n\n"
                "Instructions:\n"
                "1. Create a single, unified draft that combines the strengths of all versions\n"
                "2. Maintain all valuable citations and academic rigor\n"
                "3. Ensure comprehensive coverage of the research question\n"
                "4. Preserve technical accuracy and detailed analysis\n"
                "5. Keep the most insightful elements from each draft"
            )
        }
    ]

    # Try models in sequence with fallbacks
    def try_model(model_func, messages):
        try:
            response = model_func(messages)
            if response and not response.startswith("[Error"):
                return response
        except Exception as e:
            root_logger.error(f"{model_func.__name__} failed: {str(e)}")
        return None

    # First try O3-mini
    result = try_model(call_o3mini, messages)
    if result:
        return result

    # Fallback to Gemini
    print("\nFalling back to Gemini for integration...")
    result = try_model(call_gemini, messages)
    if result:
        return result

    # Final fallback to DeepSeek
    print("\nFalling back to DeepSeek for integration...")
    return call_deepseek_original(messages)  # Use original DeepSeek as final fallback


async def run_research_pathway(user_query):
    """
    Updated research pathway using multiple LLMs for content generation.
    """
    try:
        print(f"Starting research pathway for query: {user_query}")
        
        # Get project manager plan with retries
        max_retries = 3
        for attempt in range(max_retries):
            try:
                pm_response = project_manager_agent(
                    "",
                    f"Outline how best to answer this question in a rigorous scientific manner: {user_query}"
                )
                if pm_response and not pm_response.startswith("[Error"):
                    print("Project manager plan completed successfully")
                    break
                raise ValueError("Invalid project manager response")
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                print(f"Project manager attempt {attempt + 1} failed: {e}")
                await asyncio.sleep(2)

        # Phase 1: Initial Content Generation with Picked References
        print("Starting iterative search process...")
        picked_references, search_history = await iterative_search_engine(
            user_query=user_query,
            manager_plan=pm_response,
            max_search_rounds=8,
            max_references=12
        )
        
        if not picked_references:
            print("No references found during search process")
            return

        store_search_results(picked_references)
        
        # Generate content using multiple models
        print("Generating content from picked references using multiple models...")
        
        drafts = {}
        
        # Try to generate each draft with fallbacks
        for model in ["gemini", "deepseek", "o3mini"]:
            try:
                draft = content_developer_agent(
                    context=None,
                    question=user_query,
                    manager_response=pm_response,
                    references_data=picked_references,
                    model_choice=model
                )
                if draft and not draft.startswith("[Error"):
                    drafts[model] = draft
                    print(f"Successfully generated {model} draft")
                else:
                    print(f"Failed to generate {model} draft")
            except Exception as e:
                root_logger.error(f"{model} draft generation failed: {str(e)}")
                continue
        
        # Ensure we have at least two drafts for integration
        if len(drafts) < 2:
            print("Insufficient valid drafts, attempting additional generation...")
            # Try one more time with default model
            try:
                backup_draft = content_developer_agent(
                    context=None,
                    question=user_query,
                    manager_response=pm_response,
                    references_data=picked_references
                )
                if backup_draft and not backup_draft.startswith("[Error"):
                    drafts["backup"] = backup_draft
            except Exception as e:
                root_logger.error(f"Backup draft generation failed: {str(e)}")
        
        # Integrate available drafts
        print(f"Integrating {len(drafts)} available drafts...")
        content = integrate_drafts(
            draft1=drafts.get("gemini", drafts.get("deepseek", "")),
            draft2=drafts.get("deepseek", drafts.get("o3mini", "")),
            draft3=drafts.get("o3mini", drafts.get("backup", "")),
            question=user_query  # Add the missing question parameter
        )
        
        if not content or content.startswith("[Error"):
            raise ValueError("Failed to generate integrated content draft")



        # Phase 2: Consider Leftover References
        print("\nEvaluating leftover references...")
        final_version = content
        all_used_references = picked_references.copy()

        # Get ALL raw search results from search history
        raw_search_results = []
        
        root_logger.debug(f"Search history contains {len(search_history.searches)} searches")
        for i, search_attempt in enumerate(search_history.searches):
            root_logger.debug(f"\nProcessing search attempt {i+1}:")
            root_logger.debug(f"Search attempt structure: {json.dumps(search_attempt, indent=2)}")
            
            if isinstance(search_attempt, dict):
                # Get ALL results from the search, not just the picked ones
                raw_result = search_attempt.get('raw_result', {})
                if isinstance(raw_result, str):
                    try:
                        raw_result = json.loads(raw_result)
                    except json.JSONDecodeError:
                        root_logger.error("Failed to parse string result as JSON")
                        raw_result = {}
                
                if isinstance(raw_result, dict) and 'results' in raw_result:
                    results = raw_result['results']
                    if isinstance(results, list):
                        raw_search_results.extend(results)
                        root_logger.debug(f"Added {len(results)} raw results from search attempt {i+1}")
            
        root_logger.debug(f"\nTotal raw results collected: {len(raw_search_results)}")
            
        additional_refs = leftover_references_evaluator(
            user_question=user_query,
            manager_plan=pm_response,
            all_search_results=raw_search_results,
            picked_references=picked_references,
            draft_content=final_version  # Pass in the first draft text
        )
        
        if additional_refs:
            print(f"Found {len(additional_refs)} potential additional references")
            
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a Research Content Evaluator analyzing if additional references would strengthen the existing text. "
                        "Would incorporating any of these references improve the text's completeness, nuance, accuracy, depth, or context? "
                        "If yes, please provide an slightly improved version of the text and reference list. If no, reply with 'THE_LEFTOVER_REFERENCES_WILL_NOT_IMPROVE_THE_TEXT'\n\n"
                        f"Current text:\n{final_version}\n\n"
                        "Additional references available:\n"
                        f"{json.dumps([{k: v for k, v in ref.items() if k in ['title', 'snippet', 'url']} for ref in additional_refs], indent=2)}\n\n"

                        "Take an inclusive approach and consider references that:\n"
                        "1. Add supporting evidence or nuance to existing points\n"
                        "2. Provide valuable context or background information\n"
                        "3. Strengthen methodological discussions\n"
                        "4. Fill gaps in current citations\n"
                        "5. Offer relevant policy or practical implications\n"
                        "6. Present alternative measurement approaches\n"
                        "7. Describe mechanisms or pathways\n"
                        "8. Offer population-specific insights\n\n"
                        "Include references if they provide ANY of these contributions:\n"
                        "- Support key relationships, even indirectly\n"
                        "- Add depth to the discussion\n"
                        "- Provide supporting examples\n"
                        "- Offer valuable contextual information\n"
                        "- Present alternative approaches or measures\n"
                        "- Describe mechanisms or mediating factors\n"
                        "- Strengthen the evidence base\n"
                        "- Highlight gaps or future directions\n\n"
                        "When revising the text:\n"
                        "1. Integrate new references naturally to expand understanding\n"
                        "2. Use them to add nuance or context to existing points\n"
                        "3. Consider creating new subsections if warranted\n"
                        "4. Maintain clear organization and flow\n\n"
                        "Return either:\n"
                        "1. A revised version of the text incorporating valuable additional references\n"
                        "2. 'THE_LEFTOVER_REFERENCES_WILL_NOT_IMPROVE_THE_TEXT' ONLY if NONE of the references "
                        "would meaningfully contribute to the text's completeness, nuance, accuracy, depth, or context"
                    )
                },
                {
                    "role": "user",
                    "content": (
                        "Would incorporating any of these references improve the text's completeness, nuance, accuracy, depth, or context? "
                        "If yes, please provide an slightly improved version of the text and reference list. If no, reply with 'THE_LEFTOVER_REFERENCES_WILL_NOT_IMPROVE_THE_TEXT'\n\n"
                        f"START OF CURRENT TEXT:\n{final_version}\n\nEND OF CURRENT TEXT\n\n"
                    )
                }
            ]
            
            leftover_evaluation = call_deepseek(messages)
            
            if "THE_LEFTOVER_REFERENCES_WILL_NOT_IMPROVE_THE_TEXT" not in leftover_evaluation:
                print("Incorporating valuable additional references...")
                final_version = leftover_evaluation
                # Merge additional references with existing ones
                for add_ref in additional_refs:
                    nurl = add_ref.get('url', '').strip().rstrip('/').replace('\\', '/').lower()
                    duplicate = next((ref for ref in all_used_references 
                                   if ref.get('url', '').strip().rstrip('/').replace('\\', '/').lower() == nurl), None)
                    if duplicate:
                        # Merge metadata fields
                        for key, value in add_ref.items():
                            if key not in duplicate or (isinstance(value, str) and 
                                                     len(value) > len(duplicate.get(key, ''))):
                                duplicate[key] = value
                        # Merge source queries
                        duplicate.setdefault("queries", [])
                        if add_ref.get("source_query") and add_ref.get("source_query") not in duplicate["queries"]:
                            duplicate["queries"].append(add_ref.get("source_query"))
                    else:
                        all_used_references.append(add_ref)
                # Update the picked references file
                store_search_results(all_used_references)
            else:
                print("Additional references would not improve the text")

        # Phase 3: Citation Evaluation for Final Version
        print("\nPerforming citation evaluation on final version...")
        search_results_like = []
        for ref in all_used_references:
            search_results_like.append({
                "query": ref.get("source_query", ""),
                "results": {"results": [ref]}
            })

        try:
            # Update to handle all returned values
            citation_check_result = await check_citations_in_content_async(
                content=final_version,
                search_results=search_results_like,
                user_query=user_query,
                call_deepseek=call_deepseek
            )
            
            # Properly unpack the returned values
            if isinstance(citation_check_result, tuple):
                if len(citation_check_result) >= 2:
                    citation_reviews = citation_check_result[0]
                    error_flag = citation_check_result[1]
                    # Handle any additional returned values if needed
                else:
                    raise ValueError("Insufficient values returned from citation check")
            else:
                citation_reviews = citation_check_result  # If not a tuple, assume it's the reviews directly

        except Exception as e:
            root_logger.error(f"Citation check failed: {e}")
            citation_reviews = {
                "citations": {},
                "summary": {
                    "total_urls": 0,
                    "successful_fetches": 0,
                    "failed_fetches": 0,
                    "accepted_citations": 0,
                    "rejected_citations": 0
                }
            }

        # Save citation checks
        try:
            with open(os.path.join(OUTPUT_FOLDER, "citations_check.json"), 'w', encoding='utf-8') as f:
                json.dump(citation_reviews, f, indent=2, ensure_ascii=False)
        except Exception as e:
            root_logger.error(f"Failed to store citation reviews: {e}")

        # Quality review
        review_context = {
            "question": user_query,
            "plan": pm_response,
            "citation_reviews": citation_reviews,
            "content": final_version
        }
        review = quality_reviewer_agent(
            json.dumps(review_context, indent=2),
            f"Assess quality and completeness of the answer to: {user_query}"
        )

        # Final revision based on citation checks and review
        final_context = {
            "question": user_query,
            "plan": pm_response,
            "citation_reviews": citation_reviews,
            "quality_review": review,
            "content": final_version
        }
        final_version = final_revision_agent(
            json.dumps(final_context, indent=2),
            f"Create the final version of the text addressing all feedback from the citation check and quality review. "
            f"Ensure the answer fully addresses: {user_query}"
        )


        # Check for missing citations
        missing_citations = validate_final_references(final_version, all_used_references)
        if missing_citations:
            root_logger.warning(f"Missing citations in final reference list: {missing_citations}")
            # Try to fetch missing citations
            async with aiohttp.ClientSession() as session:
                semaphore = asyncio.Semaphore(5)  # Limit concurrent requests
                for citation in missing_citations:
                    try:
                        if citation.startswith('10.'):  # DOI
                            text, error, metadata = await fetch_with_crossref(citation)
                        else:  # URL
                            text, error, metadata = await fetch_full_text_async(
                                citation, session, semaphore)
                        if not error and metadata:
                            # Merge the new reference using our established merge logic
                            nurl = citation.strip().rstrip('/').replace('\\', '/').lower()
                            duplicate = next((ref for ref in all_used_references 
                                           if ref.get('url', '').strip().rstrip('/').replace('\\', '/').lower() == nurl), None)
                            if duplicate:
                                # Merge metadata fields
                                for key, value in metadata.items():
                                    if key not in duplicate or (isinstance(value, str) and 
                                                             len(value) > len(duplicate.get(key, ''))):
                                        duplicate[key] = value
                            else:
                                all_used_references.append(metadata)
                    except Exception as e:
                        root_logger.error(f"Failed to fetch missing citation {citation}: {e}")

        # Store outputs
        outputs = {
            "content_draft": content,
            "quality_review": review,
            "final_version": final_version,
            "project_manager": pm_response,
            "picked_references": json.dumps(all_used_references, indent=2)
        }

        for name, content_str in outputs.items():
            try:
                output_file = os.path.join(CONTENT_FOLDER, f"{name}.txt")
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(content_str)
                print(f"Saved {name} to {output_file}")
            except Exception as e:
                root_logger.error(f"Failed to save {name}: {e}")

        print("Research pathway completed successfully")
        return final_version

    except Exception as e:
        root_logger.error(f"Research pathway failed: {e}", exc_info=True)
        raise

def pydantic_to_dict(obj):
    """
    Recursively convert a Pydantic model or nested data structure 
    to pure Python primitives (dict, list, str, etc.).
    """
    if isinstance(obj, BaseModel):
        return pydantic_to_dict(obj.dict())
    elif isinstance(obj, list):
        return [pydantic_to_dict(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: pydantic_to_dict(v) for k, v in obj.items()}
    else:
        return obj




async def single_search_engine(user_query: str, manager_plan: str) -> List[dict]:
    """
    Execute a single search round for final revision purposes.
    This function uses the researcher_agent to generate a search query, calls the search tool once,
    processes the returned results using the search_result_picker_agent, and returns the list of new
    (relevant) reference objects.
    """
    # Generate a search query
    root_logger.info("\n[Single Search] Generating search query for final revision...")
    first_query_response = researcher_agent(context="", question=user_query, manager_plan=manager_plan)
    try:
        query_data = json.loads(first_query_response)
        current_query = query_data.get('query', first_query_response)
    except json.JSONDecodeError:
        current_query = first_query_response.strip().replace('```json', '').replace('```', '').strip()
    
    root_logger.info(f"[Single Search] Query: {current_query}")
    
    # Execute a single search attempt with retry logic
    raw_result = {}
    max_retries = 3
    for attempt in range(max_retries):
        try:
            result_data = search_tool._run(current_query)
            raw_result = pydantic_to_dict(result_data)
            if isinstance(raw_result, str):
                try:
                    raw_result = json.loads(raw_result)
                except json.JSONDecodeError:
                    root_logger.error("[Single Search] Failed to parse string result as JSON")
                    raw_result = {}
            break  # Exit loop on success
        except Exception as e:
            root_logger.error(f"[Single Search] Search attempt {attempt + 1} failed: {str(e)}")
            if attempt == max_retries - 1:
                root_logger.error("[Single Search] All search attempts failed")
                break
            await asyncio.sleep(2)
    
    # Pick relevant results using the search result picker
    picked_results = search_result_picker_agent(user_query, raw_result)
    relevant_list = []
    if picked_results.get("status") != "NO_RELEVANT_SEARCH_RESULTS":
        picked_urls = set(result["url"] for result in picked_results.get("relevant_results", []))
        if "results" in raw_result:
            results = raw_result["results"]
            if isinstance(results, dict) and "results" in results:
                results = results["results"]
            for result in results:
                if result.get("url") in picked_urls:
                    relevant_list.append(result)
    else:
        root_logger.warning("[Single Search] No relevant results found.")
    
    return relevant_list



def final_revision_router_agent(final_text: str, user_question: str) -> dict:
    """
    Sends the final text and research question to the LLM with instructions
    to decide whether additional changes are needed.
    Expects a JSON response with:
      - action: one of NO_ADDITIONAL_CHANGES_NEEDED, EXTRA_SEARCH_NEEDED, ONLY_REVISION_NEEDED, or FETCH_URL_NEEDED
      - reason: a short explanation
      - urls: list of URLs needing metadata fetch (only for FETCH_URL_NEEDED action)
    """
    messages = [
        {
            "role": "system",
            "content": (
                "You are a final revision router evaluating both content quality and academic rigor. "
                "Review the text for:\n\n"
                "1. Reference Completeness:\n"
                "   - Every citation MUST include all required metadata:\n"
                "     * Author names\n"
                "     * Publication year\n"
                "     * Title\n"
                "     * Journal/source\n"
                "     * URL or DOI\n"
                "   - Flag ANY citation that:\n"
                "     * Contains placeholders like [URL needed] or [needs citation]\n"
                "     * Is missing author information\n"
                "     * Lacks publication year\n"
                "     * Has incomplete source details\n"
                "   - Return FETCH_URL_NEEDED if ANY citations are incomplete\n\n"
                "2. Answer Completeness:\n"
                "   - Fully addresses all aspects of the research question\n"
                "   - Provides sufficient depth and detail\n"
                "   - Includes relevant examples and evidence\n"
                "   - Covers important counterpoints or limitations\n\n"
                "3. Academic Quality:\n"
                "   - Uses peer-reviewed academic sources\n"
                "   - Provides complete reference information\n"
                "   - Avoids non-academic sources\n\n"
                "4. Clarity and Structure:\n"
                "   - Clear organization and flow\n"
                "   - Well-supported arguments\n"
                "   - Appropriate technical depth\n\n"
                "Choose FETCH_URL_NEEDED if:\n"
                "- Citations are missing author information\n"
                "- Reference metadata is incomplete\n"
                "- URLs need author/publication details\n"
                "Return the list of URLs needing metadata fetch\n\n"
                "Choose EXTRA_SEARCH_NEEDED if:\n"
                "- Important aspects of the question remain unaddressed\n"
                "- Key evidence or examples are missing\n"
                "- Non-academic sources need replacement\n"
                "- Additional perspectives are needed\n\n"
                "Choose ONLY_REVISION_NEEDED if:\n"
                "- Content is complete but needs reorganization\n"
                "- Arguments need better structuring\n"
                "- Citations need better integration\n"
                "- Writing needs clarity improvements\n\n"
                "Choose NO_ADDITIONAL_CHANGES_NEEDED if:\n"
                "- Question is fully addressed\n"
                "- Arguments are well-supported\n"
                "- Academic sources are properly used\n"
                "- Content is clear and well-organized\n"
                "- All references have complete metadata"
            )
        },
        {
            "role": "user",
            "content": (
                f"Research Question: {user_question}\n\n"
                f"Final Text:\n{final_text}\n\n"
                "Return a JSON response with:\n"
                "  - action: one of NO_ADDITIONAL_CHANGES_NEEDED, EXTRA_SEARCH_NEEDED, ONLY_REVISION_NEEDED, or FETCH_URL_NEEDED\n"
                "  - reason: a short explanation\n"
                "  - urls: list of URLs needing metadata fetch (only for FETCH_URL_NEEDED action)"
            )
        }
    ]
    response = call_deepseek(messages, fast_mode=True)
    parsed = extract_json(response)
    if parsed and isinstance(parsed, list) and isinstance(parsed[0], dict):
        result = parsed[0]
        # Ensure urls field exists for FETCH_URL_NEEDED action
        if result.get("action") == "FETCH_URL_NEEDED" and "urls" not in result:
            result["urls"] = []
        return result
    else:
        return {
            "action": "EXTRA_SEARCH_NEEDED",
            "reason": "Unable to verify content completeness and quality - additional review needed.",
            "urls": []
        }

def validate_final_references(final_text: str, final_reference_list: List[dict] = None) -> List[str]:
    """
    Validate that all citations in the final text are complete and present in the reference list.
    Returns a list of problematic citations.
    """
    if final_reference_list is None:
        final_reference_list = get_current_references()
    
    problems = []
    
    # Check for placeholder markers with refined handling for [needs citation]
    placeholder_pattern = r'\[(URL|citation|reference|source|author|year)\s*(needed|missing|incomplete|unknown)\]'
    placeholders = re.finditer(placeholder_pattern, final_text, re.IGNORECASE)
    for match in placeholders:
        placeholder_text = match.group(0)
        if "needs citation" in placeholder_text.lower():
            problems.append("NEEDS_CITATION_PLACEHOLDER")
        else:
            problems.append(f"Placeholder found: {placeholder_text}")
    
    # Extract URLs from text
    url_pattern = r'https?://[^\s<>"]+|www\.[^\s<>"]+|doi\.org/[^\s<>"]+|dx\.doi\.org/[^\s<>"]+'
    extracted_urls = set(re.findall(url_pattern, final_text))
    
    # Extract DOIs from text
    doi_pattern = r'\b(10\.\d{4,9}/[-._;()/:A-Z0-9]+)\b'
    extracted_dois = set(re.findall(doi_pattern, final_text, re.I))
    
    # Get reference identifiers
    reference_urls = {
        ref.get('url', '').strip().rstrip('/').replace('\\', '/').lower(): ref
        for ref in final_reference_list
    }
    reference_dois = {
        ref.get('doi', '').lower(): ref
        for ref in final_reference_list if ref.get('doi')
    }
    
    # Check for missing or incomplete references among URLs
    for url in extracted_urls:
        normalized_url = url.strip().lower().rstrip('/')
        ref = reference_urls.get(normalized_url)
        if not ref:
            problems.append(f"Missing reference for URL: {url}")
        elif not all(ref.get(field) for field in ['authors', 'year', 'title']):
            problems.append(f"Incomplete metadata for URL: {url}")
    
    # Check for missing or incomplete references among DOIs
    for doi in extracted_dois:
        normalized_doi = doi.lower()
        ref = reference_dois.get(normalized_doi)
        if not ref:
            problems.append(f"Missing reference for DOI: {doi}")
        elif not all(ref.get(field) for field in ['authors', 'year', 'title']):
            problems.append(f"Incomplete metadata for DOI: {doi}")
    
    return problems

async def handle_citation_placeholders(content: str, all_used_references: List[dict], user_query: str) -> str:
    """
    Processes the content that contains [needs citation] placeholders by attempting to integrate
    appropriate citations from the available references.
    """
    messages = [
        {
            "role": "system",
            "content": "You are a Citation Integration Specialist. Your task is to replace [needs citation] placeholders in the text with appropriate citation details extracted from the available references."
        },
        {
            "role": "user",
            "content": (
                f"Research Question: {user_query}\n\n"
                f"Available References:\n{json.dumps([{k: v for k, v in ref.items() if k in ['title', 'url']} for ref in all_used_references], indent=2)}\n\n"
                f"Content with Placeholders:\n{content}\n\n"
                "For each [needs citation] placeholder, replace it with a proper citation from the references if applicable. "
                "If no suitable reference is found, leave a note in the format [Citation needed: specific topic]."
            )
        }
    ]
    
    try:
        # Try multiple models with fallback
        for model_func in [call_o3mini, call_gemini, call_deepseek]:
            try:
                result = model_func(messages)
                if result and not result.startswith("[Error"):
                    return result
            except Exception as e:
                root_logger.error(f"{model_func.__name__} failed in handle_citation_placeholders: {e}")
                continue
        return content  # Return original if all attempts fail
    except Exception as e:
        root_logger.error(f"handle_citation_placeholders encountered an error: {e}")
        return content

async def final_revision_controller(
    user_query: str,
    current_final_text: str,
    original_draft: str,
    max_iterations: int = 3
) -> str:
    """
    Iteratively refines the text while preserving domain-specific references.
    
    Args:
        user_query: The original research question
        current_final_text: The current version of the text
        original_draft: The first final draft with domain references
        max_iterations: Maximum number of revision iterations
    
    Returns:
        The final revised text with preserved domain references
    """
    current_text = current_final_text
    iteration = 0
    attempted_urls = set()  # Track URLs we've already tried to fetch

    while iteration < max_iterations:
        iteration += 1
        
        # First check with router before citation validation
        router_response = final_revision_router_agent(current_text, user_query)
        action = router_response.get("action", "NO_ADDITIONAL_CHANGES_NEEDED")
        reason = router_response.get("reason", "")
        print(f"Final revision iteration {iteration}: Action={action}, Reason={reason}")

        # Before handling other actions, if placeholders are present, try to process them
        if "[needs citation]" in current_text.lower():
            print("Detected [needs citation] placeholders. Attempting to integrate appropriate citations...")
            current_text = await handle_citation_placeholders(
                content=current_text,
                all_used_references=get_current_references(),
                user_query=user_query
            )
        
        # Revalidate citations after placeholder handling
        citation_problems = validate_final_references(current_text)
        if action == "NO_ADDITIONAL_CHANGES_NEEDED":
            if citation_problems:
                # If the only issues are citation placeholders, handle them
                if all(problem == "NEEDS_CITATION_PLACEHOLDER" for problem in citation_problems):
                    print("Only [needs citation] placeholders detected. Reprocessing them...")
                    current_text = await handle_citation_placeholders(
                        content=current_text,
                        all_used_references=get_current_references(),
                        user_query=user_query
                    )
                    citation_problems = validate_final_references(current_text)
                    if not citation_problems:
                        break
                    else:
                        action = "FETCH_URL_NEEDED"
                        router_response = {
                            "action": "FETCH_URL_NEEDED",
                            "reason": "Incomplete citations detected",
                            "urls": citation_problems
                        }
                else:
                    action = "FETCH_URL_NEEDED"
                    router_response = {
                        "action": "FETCH_URL_NEEDED",
                        "reason": "Incomplete citations detected",
                        "urls": citation_problems
                    }
            else:
                break

        if action == "FETCH_URL_NEEDED":
            urls = router_response.get("urls", [])
            # Filter out placeholder markers for URL fetching purposes
            new_urls = [url for url in urls if url != "NEEDS_CITATION_PLACEHOLDER" and url not in attempted_urls]
            
            # If placeholders still exist, try handling them
            if "NEEDS_CITATION_PLACEHOLDER" in urls:
                print("Handling [needs citation] placeholders separately...")
                current_text = await handle_citation_placeholders(
                    content=current_text,
                    all_used_references=get_current_references(),
                    user_query=user_query
                )
                # Revalidate citations after handling placeholders
                citation_problems = validate_final_references(current_text)
                if not citation_problems:
                    break
                else:
                    new_urls = [url for url in citation_problems if url != "NEEDS_CITATION_PLACEHOLDER" and url not in attempted_urls]

            if new_urls:
                print(f"Fetching metadata for {len(new_urls)} references...")
                complete_refs = await fetch_missing_metadata(new_urls)
                attempted_urls.update(new_urls)  # Mark these URLs as attempted
                
                if complete_refs:
                    # Provide focused context for reference updates
                    revision_context = {
                        "current_text": current_text,
                        "reference_updates": complete_refs,
                        "failed_urls": list(attempted_urls - set(ref.get('url') for ref in complete_refs)),
                        "instructions": "Update ONLY the reference metadata while preserving all content and arguments. Add author names, publication years, and other citation details where missing."
                    }
                    current_text = final_revision_agent(
                        context=json.dumps(revision_context, indent=2),
                        question="Update reference metadata while preserving all content and arguments"
                    )
                    print("Updated reference metadata in text")
                else:
                    print("No metadata could be extracted")
                    # Switch to revision mode if no metadata could be extracted
                    action = "ONLY_REVISION_NEEDED"
            else:
                print("All URLs already attempted or only placeholders remain, switching to revision mode")
                action = "ONLY_REVISION_NEEDED"

        elif action == "EXTRA_SEARCH_NEEDED":
            # Generate new search query using current context
            new_search_query = researcher_agent(
                context=current_text,
                question=user_query,
                manager_plan="Extra references to refine final text"
            )
            new_refs = await single_search_engine(new_search_query, manager_plan="")

            if new_refs:
                # Integrate new references while preserving original domain context
                revision_context = {
                    "original_draft": original_draft,
                    "current_text": current_text,
                    "new_references": new_refs
                }
                current_text = content_developer_agent(
                    context=json.dumps(revision_context, indent=2),
                    question=user_query,
                    manager_response="Integrate new references while preserving domain specifics",
                    references_data=new_refs
                )
            else:
                # Fallback to revision if no new references found
                revision_context = {
                    "original_draft": original_draft,
                    "current_text": current_text,
                    "reason": reason
                }
                current_text = final_revision_agent(
                    context=json.dumps(revision_context, indent=2),
                    question=f"Revise while preserving domain references: {user_query}"
                )

        elif action == "ONLY_REVISION_NEEDED":
            revision_context = {
                "original_draft": original_draft,
                "current_text": current_text,
                "reason": reason,
                "failed_urls": list(attempted_urls),
                "instructions": "Revise the text to handle incomplete citations by either using available metadata, marking claims as [needs citation], or removing unsupported claims if necessary."
            }
            current_text = final_revision_agent(
                context=json.dumps(revision_context, indent=2),
                question=f"Revise while preserving domain references: {user_query}"
            )
        else:
            break

        # Post-processing check
        final_problems = validate_final_references(current_text)
        if final_problems:
            if iteration == max_iterations:
                # Remove problematic citations on final iteration
                revision_context = {
                    "original_draft": original_draft,
                    "current_text": current_text,
                    "problems": final_problems,
                    "failed_urls": list(attempted_urls),
                    "instructions": "Remove any incomplete citations and their associated claims. Ensure all remaining references are complete."
                }
                current_text = final_revision_agent(
                    context=json.dumps(revision_context, indent=2),
                    question=f"Create final version without incomplete citations for: {user_query}"
                )
            else:
                # Try another iteration
                continue

    # Add final validation check here
    print("\nPerforming final validation check...")
    final_router_response = final_revision_router_agent(current_text, user_query)
    final_action = final_router_response.get("action", "NO_ADDITIONAL_CHANGES_NEEDED")
    final_reason = final_router_response.get("reason", "")
    
    print(f"Final validation result: {final_action}")
    print(f"Reason: {final_reason}")

    # If changes are still needed and we haven't hit max iterations, do one more revision cycle
    if final_action != "NO_ADDITIONAL_CHANGES_NEEDED" and iteration < max_iterations:
        print("\nRunning final cleanup cycle...")
        revision_context = {
            "original_draft": original_draft,
            "current_text": current_text,
            "reason": final_reason,
            "failed_urls": list(attempted_urls),
            "instructions": "Final cleanup: address remaining issues while preserving key content and complete references."
        }
        current_text = final_revision_agent(
            context=json.dumps(revision_context, indent=2),
            question=f"Final cleanup for: {user_query}"
        )

    return current_text

async def main():
    """
    Simplified main function that runs a single research pathway and preserves domain references
    during final revision.
    """
    user_query = input("Enter your research question: ")

    # Initialize folders and logging
    setup_folders()
    root_logger.info("\n=== RUNNING SINGLE-ROUND RESEARCH PATHWAY ===")

    # Run single research pathway
    final_version = await run_research_pathway(user_query)
    if not final_version:
        root_logger.error("Research pathway failed to produce output")
        return

    # Pass to final revision loop with original draft preserved
    revised_final_version = await final_revision_controller(
        user_query=user_query,
        current_final_text=final_version,
        original_draft=final_version,  # Preserve domain references
        max_iterations=3
    )

    # Log results
    root_logger.info(f"\n[Final Revised Version]\n{revised_final_version}")
    root_logger.info(f"\nOutputs have been saved to: {OUTPUT_FOLDER}")

    # Save final version
    try:
        final_path = os.path.join(OUTPUT_FOLDER, "content", "final_version.txt")
        with open(final_path, 'w', encoding='utf-8') as f:
            f.write(revised_final_version)
        print(f"Final version saved to: {final_path}")
    except Exception as e:
        root_logger.error(f"Failed to save final version: {e}")

async def fetch_missing_metadata(urls: List[str]) -> List[dict]:
    """
    Fetch metadata for URLs that have incomplete citation information.
    
    Args:
        urls: List of URLs or citation identifiers needing metadata
    
    Returns:
        List of reference objects with complete metadata
    """
    complete_refs = []
    
    # Load existing references from the session file with corrected path
    existing_refs = []
    try:
        refs_path = os.path.join(OUTPUT_FOLDER, "search_results", "session_picked_references.json")
        with open(refs_path, 'r', encoding='utf-8') as f:
            existing_refs = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        root_logger.warning(f"Could not load existing references: {e}")
        existing_refs = []
    
    # Clean up URLs that might have error message prefixes
    cleaned_urls = []
    for url in urls:
        if isinstance(url, str):
            if url.startswith("Missing reference for URL:"):
                url = url.replace("Missing reference for URL:", "").strip()
            cleaned_urls.append(url)
    
    async with aiohttp.ClientSession() as session:
        semaphore = asyncio.Semaphore(5)  # Limit concurrent requests
        
        for url in cleaned_urls:
            try:
                if url.startswith('10.'):  # DOI
                    # Try academic APIs first
                    for method in [fetch_with_google_scholar, fetch_with_semantic_scholar, fetch_with_crossref]:
                        text, error, metadata = await method(url)
                        if not error and metadata:
                            complete_refs.append(metadata)
                            break
                else:  # URL
                    text, error, metadata = await fetch_full_text_async(url, session, semaphore)
                    if not error and metadata:
                        # Extract additional metadata from content if available
                        content_metadata = await extract_metadata_from_content(text)
                        metadata.update(content_metadata)
                        complete_refs.append(metadata)
                
                if complete_refs:
                    # Update existing references
                    for ref in complete_refs:
                        # Check if reference already exists
                        existing = next(
                            (r for r in existing_refs 
                             if r.get('url') == ref.get('url') or r.get('doi') == ref.get('doi')), 
                            None
                        )
                        if existing:
                            # Update existing reference with any new metadata
                            for key, value in ref.items():
                                if key not in existing or (
                                    isinstance(value, str) and 
                                    len(value) > len(existing.get(key, ''))
                                ):
                                    existing[key] = value
                        else:
                            existing_refs.append(ref)
                    
                    # Store updated references
                    store_search_results(existing_refs)
                
            except Exception as e:
                root_logger.error(f"Error fetching metadata for {url}: {str(e)}")
                continue
    
    return complete_refs

def get_current_references() -> List[dict]:
    """
    Get the current list of references from the session file.
    """
    try:
        refs_path = os.path.join(OUTPUT_FOLDER, "search_results", "session_picked_references.json")
        with open(refs_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        root_logger.warning(f"Could not load references: {e}")
        return []

if __name__ == "__main__":
    asyncio.run(main())


