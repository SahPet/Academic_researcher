import aiohttp
import json
import asyncio
import logging
from bs4 import BeautifulSoup
from typing import Dict, Any, Set
from asyncio import Semaphore
from datetime import datetime, timedelta
from collections import deque
from dotenv import load_dotenv
import os

# 1. Add Error Classes at the top
class SearchError(Exception):
    """Base exception for search errors"""
    pass

class APIError(SearchError):
    """API-related errors"""
    pass

class MetadataError(SearchError):
    """Metadata fetch errors"""
    pass

# 2. Add Rate Limiter Class
class RateLimiter:
    def __init__(self, max_calls: int, period: int):
        self.max_calls = max_calls
        self.period = period
        self.calls = deque()
        self.sem = Semaphore(max_calls)

    async def acquire(self):
        now = datetime.now()
        # Remove old calls
        while self.calls and now - self.calls[0] > timedelta(seconds=self.period):
            self.calls.popleft()
            self.sem.release()
        await self.sem.acquire()
        self.calls.append(now)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class SerperSearchTool:
    def __init__(self):
        self.name = "Search Engine Tool"
        self.description = "Search the internet with enhanced metadata using Serper API"
        self.rate_limiter = RateLimiter(max_calls=10, period=60)
        self.request_semaphore = Semaphore(5)
        self.logger = logging.getLogger("serper_tool")
        
        # Load API key from .env file
        load_dotenv()
        
        # Get API key from environment variables
        self.api_key = os.getenv('SERPER_API_KEY')
        if not self.api_key:
            raise ValueError("SERPER_API_KEY not found in environment variables")

        self.EXCLUDE_TAGS = {
            # Technical/device tags
            'viewport', 'theme-color', 'applicable-device', 'format-detection',
            'handheldfriendly', 'mobileoptimized', 'apple-mobile-web-app-capable',
            'apple-mobile-web-app-status-bar-style', 'msapplication-tilecolor',
            'msapplication-config', 'msapplication-tileimage', 'application-name',
            
            # Verification tags
            '360-site-verification', 'google-site-verification', 'yandex-verification',
            'msvalidate.01', 'p:domain_verify', 'facebook-domain-verification',
            
            # Analytics and tracking
            'generator', 'analytics', 'google-analytics', 'ga-tracking-id',
            'facebook-pixel-id', 'amplitude-session-id',
            
            # Social media non-content
            'twitter:card', 'twitter:site', 'twitter:creator', 'twitter:app:id',
            'fb:app_id', 'og:type', 'og:site_name',
            
            # Technical SEO/service tags
            'referrer', 'robots', 'googlebot', 'bingbot', 'format', 
            'rating', 'theme', 'color-scheme', 'next-head-count',
            
            # Internal identifiers
            'ncbi_pinger_gtm_track', 'ncbi_phid', 'ncbi_pdid', 'ncbi_app',
            'access_endpoint', 'sso-service', 'log_category', 'log_op',
            'log_source_db', 'uid', 'ncbi_uid', 'log_displayeduids',
            'log_icons_present', 'size', 'access', 'optimize_experiments',
            
            # Browser/device specific
            'parsely-type', 'parsely-post-id', 'parsely-section', 'parsely-tags',
            'wt.cg_s', 'wt.z_cg_type', 'wt.page_categorisation', 'wt.z_bandiera_abtest',

            # Social media metadata
            'og:url', 'og:type', 'og:image', 'og:image:width', 'og:image:height',
            'twitter:url', 'twitter:image', 'twitter:player', 'twitter:player:width',
            'twitter:player:height', 'twitter:app:name:iphone', 'twitter:app:id:iphone',
            'twitter:app:name:ipad', 'twitter:app:id:ipad', 'twitter:app:url:iphone',
            'twitter:app:url:ipad', 'twitter:app:name:googleplay', 'twitter:app:id:googleplay',
            'twitter:app:url:googleplay',
            
            # App-related metadata
            'al:ios:app_store_id', 'al:ios:app_name', 'al:ios:url', 'al:android:url',
            'al:web:url', 'al:android:app_name', 'al:android:package',
            
            # Media-specific
            'og:video:url', 'og:video:secure_url', 'og:video:type', 'og:video:width',
            'og:video:height',
            
            # GitHub specific tags
            'route-pattern', 'route-controller', 'route-action', 'current-catalog-service-hash',
            'request-id', 'html-safe-nonce', 'visitor-payload', 'visitor-hmac',
            'hovercard-subject-tag', 'github-keyboard-shortcuts', 'octolytics-url',
            'analytics-location', 'hostname', 'turbo-cache-control', 'go-import',
            'octolytics-dimension-user_id', 'octolytics-dimension-user_login',
            'octolytics-dimension-repository_id', 'octolytics-dimension-repository_nwo',
            'octolytics-dimension-repository_public', 'octolytics-dimension-repository_is_fork',
            'turbo-body-classes', 'browser-stats-url', 'browser-errors-url',
            
            # Additional analytics and tracking
            'visitor-payload', 'visitor-hmac', 'analytics-location',
            
            # Mobile and app-related additions
            'apple-itunes-app', 'mobile-web-app-capable',
            
            # Image dimensions and technical details
            'image:width', 'image:height', 'image:alt',
            
            # Additional social media and sharing
            'twitter:description',
            
            # Browser and technical metadata
            'turbo-cache-control', 'browser-stats-url', 'browser-errors-url',
            'html-safe-nonce', 'request-id'
        }

    async def fetch_page_metadata(self, session: aiohttp.ClientSession, url: str) -> Dict[str, Any]:
        """Fetch metadata from webpage, handling different encodings."""
        try:
            async with session.get(url, allow_redirects=True, timeout=15) as response:
                if response.status != 200:
                    return {}
                    
                # First try to get the content-type header
                content_type = response.headers.get('content-type', '').lower()
                
                # Get the raw content with timeout
                try:
                    raw_content = await asyncio.wait_for(response.read(), timeout=10)
                except asyncio.TimeoutError:
                    logger.warning(f"Content read timeout for {url}")
                    return {}
                
                # Try to detect encoding from content-type header
                encoding = None
                if 'charset=' in content_type:
                    encoding = content_type.split('charset=')[-1].strip()
                
                # Try different encodings in order of likelihood
                encodings_to_try = [
                    encoding,
                    'utf-8',
                    'latin1',
                    'iso-8859-1',
                    'cp1252',
                    'ascii'
                ]
                
                html = None
                for enc in encodings_to_try:
                    if not enc:
                        continue
                    try:
                        html = raw_content.decode(enc)
                        break
                    except (UnicodeDecodeError, LookupError):
                        continue
                
                if html is None:
                    # If all decodings fail, try with errors='replace'
                    html = raw_content.decode('utf-8', errors='replace')
                
                soup = BeautifulSoup(html, 'html.parser')
                
                metadata = {}
                seen_content = set()
                
                # Priority order for metadata prefixes
                prefix_priority = ['dc.', 'citation_', 'prism.', 'article:', 'og:', '']
                meta_tags = soup.find_all('meta')
                
                for prefix in prefix_priority:
                    for tag in meta_tags:
                        name = tag.get('name', '').lower()
                        if not name:
                            name = tag.get('property', '').lower()
                        content = tag.get('content', '').strip()
                        
                        if not name or not content or content in seen_content:
                            continue
                            
                        if name.startswith(prefix):
                            if not prefix and name in self.EXCLUDE_TAGS:
                                continue
                                
                            base_name = name.replace(prefix, '')
                            
                            if not base_name:
                                continue
                                
                            if base_name in ['title', 'description']:
                                if len(content) > 10:
                                    metadata[base_name] = content
                                    seen_content.add(content)
                            elif base_name not in metadata:
                                if any(id_key in base_name.lower() for id_key in ['doi', 'issn', 'isbn', 'pmid', 'id']):
                                    metadata[base_name] = content
                                elif len(content) > 3:
                                    metadata[base_name] = content
                                seen_content.add(content)
                
                if 'title' not in metadata:
                    title_tag = soup.find('title')
                    if title_tag and title_tag.string:
                        title_content = title_tag.string.strip()
                        if len(title_content) > 10 and title_content not in seen_content:
                            metadata['title'] = title_content
                            
                return metadata
                
        except asyncio.TimeoutError:
            logger.warning(f"Timeout fetching metadata for {url}")
            return {}
        except Exception as e:
            logger.error(f"Error fetching metadata for {url}: {str(e)}")
            return {}

    async def _async_search(self, query: str, retry_count: int = 0) -> str:
        """Enhanced search with robust error handling and metadata"""
        try:
            # Validate query
            if not query.strip():
                return json.dumps({
                    "query": query,
                    "error": "Empty query",
                    "retryable": False,
                    "results": []
                })

            # Acquire rate limit slot
            await self.rate_limiter.acquire()
            
            async with aiohttp.ClientSession() as session:
                try:
                    # API call with timeout
                    async with session.post(
                        "https://google.serper.dev/search",
                        headers={"X-API-KEY": self.api_key},
                        json={"q": query, "num": 5},
                        timeout=30
                    ) as response:
                        
                        # Handle HTTP errors
                        if response.status == 429 and retry_count < 3:  # Rate limit with retry
                            self.logger.warning(f"Rate limit hit for query: {query}")
                            await asyncio.sleep(2 * (retry_count + 1))
                            return await self._async_search(query, retry_count + 1)
                            
                        if response.status != 200:
                            error_msg = f"API Error {response.status}"
                            self.logger.error(f"{error_msg} for query: {query}")
                            raise APIError(error_msg)
                            
                        # Parse JSON response
                        try:
                            results = await response.json()
                        except json.JSONDecodeError:
                            raise APIError("Invalid JSON response")
                            
                        # Validate results structure
                        if "organic" not in results:
                            raise APIError("No organic results in response")
                            
                        # Process results
                        enhanced_results = []
                        tasks = []
                        
                        async def fetch_metadata(url: str) -> Dict:
                            """Fetch metadata with concurrency control"""
                            async with self.request_semaphore:
                                try:
                                    return await self.fetch_page_metadata(session, url)
                                except Exception as e:
                                    self.logger.warning(f"Metadata fetch failed for {url}: {str(e)}")
                                    return {"error": str(e)}
                        
                        # Create base results
                        for result in results["organic"]:
                            base_entry = {
                                "title": result.get("title", ""),
                                "url": result.get("link", ""),
                                "snippet": result.get("snippet", ""),
                                "position": result.get("position", -1)
                            }
                            enhanced_results.append(base_entry)
                            
                            if base_entry["url"]:
                                tasks.append(
                                    (base_entry, fetch_metadata(base_entry["url"]))
                                )
                        
                        # Add metadata concurrently
                        for entry, task in tasks:
                            try:
                                metadata = await task
                                if metadata:
                                    entry["metadata"] = metadata
                            except Exception as e:
                                entry["metadata_error"] = str(e)
                        
                        return json.dumps({
                            "query": query,
                            "results": enhanced_results,
                            "total": len(enhanced_results),
                            "related_queries": results.get("relatedSearches", []),
                            "success": True
                        }, indent=2)
                        
                except APIError as e:
                    self.logger.error(f"Search failed for '{query}': {str(e)}")
                    return json.dumps({
                        "query": query,
                        "error": str(e),
                        "retryable": True if "429" in str(e) else False
                    })
                    
                except asyncio.TimeoutError:
                    self.logger.error(f"Search timeout for '{query}'")
                    if retry_count < 3:
                        return await self._async_search(query, retry_count + 1)
                    return json.dumps({
                        "query": query,
                        "error": "Search timeout after retries",
                        "retryable": False
                    })
                    
                except Exception as e:
                    self.logger.error(f"Unexpected error during search: {str(e)}")
                    return json.dumps({
                        "query": query,
                        "error": f"Unexpected error: {str(e)}",
                        "retryable": False
                    })
                    
        except Exception as e:
            self.logger.critical(f"Critical search failure: {str(e)}")
            return json.dumps({
                "query": query,
                "error": f"Critical system error: {str(e)}",
                "retryable": False
            })

    def _run(self, query: str) -> str:
        try:
            # Get the current event loop if it exists, or create a new one
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Run the search
            if loop.is_running():
                # If we're already in an event loop, create a new one in a thread
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(lambda: asyncio.new_event_loop().run_until_complete(self._async_search(query)))
                    result = future.result()
            else:
                # If no event loop is running, use this one
                result = loop.run_until_complete(self._async_search(query))
            
            return result
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            return json.dumps({"error": str(e)})