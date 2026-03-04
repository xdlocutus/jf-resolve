"""Stream resolution API routes"""

import asyncio
import hashlib
import sys
import traceback
import uuid
from datetime import datetime
from typing import Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import RedirectResponse, StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from ..config import settings
from ..database import get_db
from ..services.failover_manager import FailoverManager
from ..services.library_service import LibraryService
from ..services.log_service import log_service
from ..services.settings_manager import SettingsManager
from ..services.stremio_service import StremioService
from ..services.tmdb_service import TMDBService

router = APIRouter(prefix="/api/stream", tags=["stream"])

# In-memory store for resolved stream URLs (session-based)
# Maps session_id -> {"url": stream_url, "created": timestamp}
_stream_sessions: dict = {}

# Session timeout in seconds (1 hour)
SESSION_TIMEOUT = 3600

# Maximum concurrent streams (configurable)
MAX_CONCURRENT_STREAMS = 10

# Track active stream count
_active_streams = 0
_stream_lock = asyncio.Lock()

# Shared secret for internal API calls between servers
# Use getattr to avoid issues at import time
_internal_api_secret = getattr(settings, 'INTERNAL_API_SECRET', 'jf-resolve-internal-2024')


def generate_session_id(media_type: str, tmdb_id: int, quality: str, season: int = None, episode: int = None) -> str:
    """Generate a unique session ID for a stream"""
    data = f"{media_type}:{tmdb_id}:{quality}:{season}:{episode}:{uuid.uuid4().hex[:8]}"
    return hashlib.sha256(data.encode()).hexdigest()[:16]


def cleanup_expired_sessions() -> int:
    """Remove expired sessions from memory"""
    now = datetime.utcnow().timestamp()
    expired = [
        sid for sid, data in _stream_sessions.items()
        if now - data["created"] > SESSION_TIMEOUT
    ]
    for sid in expired:
        del _stream_sessions[sid]
    return len(expired)


@router.get("/resolve/{media_type}/{tmdb_id}")
async def resolve_stream(
    media_type: str,
    tmdb_id: int,
    quality: str = Query("1080p"),
    season: Optional[int] = Query(None),
    episode: Optional[int] = Query(None),
    index: int = Query(0),
    imdb_id: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
):
    """
    Resolve stream URL with failover
    Returns 302 redirect to actual stream URL from Stremio manifest
    """
    log_service.info(
        f"Stream resolve request: {media_type}/{tmdb_id} quality={quality} "
        f"index={index} imdb_id={imdb_id} season={season} episode={episode}"
    )

    if media_type not in ["movie", "tv"]:
        raise HTTPException(status_code=400, detail="Invalid media type")

    if media_type == "tv" and (season is None or episode is None):
        raise HTTPException(
            status_code=400, detail="Season and episode required for TV shows"
        )

    settings = SettingsManager(db)
    await settings.load_cache()

    tmdb = None
    api_key = await settings.get("tmdb_api_key")

    manifest_url = await settings.get("stremio_manifest_url")
    if not manifest_url:
        raise HTTPException(
            status_code=500, detail="Stremio manifest URL not configured"
        )

    stremio = StremioService(manifest_url)

    failover = FailoverManager(db)

    try:
        if media_type == "movie":
            state_key = f"movie:{tmdb_id}"
        else:
            state_key = f"tv:{tmdb_id}:{season}:{episode}"

        grace_seconds = await settings.get("failover_grace_seconds", 45)
        reset_seconds = await settings.get("failover_window_seconds", 120)

        state = await failover.get_state(state_key)

        should_increment, use_index = failover.should_failover(
            state, grace_seconds, reset_seconds
        )

        now = datetime.utcnow()
        if state.first_attempt is None:
            state.first_attempt = now
        state.last_attempt = now

        if should_increment:
            state.current_index = use_index
            state.attempt_count += 1
        else:
            use_index = state.current_index

        await failover.update_state(state)

        if not imdb_id:
            if not api_key:
                raise HTTPException(
                    status_code=500, detail="TMDB API key not configured"
                )
            tmdb = TMDBService(api_key)
            library = LibraryService(db, tmdb, settings)
            imdb_id = await library.get_or_fetch_imdb_id(tmdb_id, media_type)

        if not imdb_id:
            log_service.error(f"No IMDB ID found for {media_type}:{tmdb_id}")
            raise HTTPException(status_code=404, detail="IMDB ID not found")

        if media_type == "movie":
            streams = await stremio.get_movie_streams(imdb_id)
        else:
            streams = await stremio.get_episode_streams(imdb_id, season, episode)

        if not streams:
            log_service.error(
                f"Stremio addon returned zero streams for {state_key} (IMDb: {imdb_id})"
            )
            raise HTTPException(
                status_code=404, detail="No streams available from addon"
            )

        fallback_enabled = await settings.get("quality_fallback_enabled", True)
        fallback_order = await settings.get(
            "quality_fallback_order", ["1080p", "720p", "4k", "480p"]
        )

        target_quality = quality
        if not quality or quality == "auto":
            target_quality = await settings.get("series_preferred_quality", "1080p")

        stream_url = await stremio.select_stream(
            streams, target_quality, use_index, fallback_enabled, fallback_order
        )

        if not stream_url:
            log_service.error(
                f"Stream selection failed for {state_key}. Quality requested: {target_quality}, "
                f"Index: {use_index}, Total streams: {len(streams)}"
            )
            available_qualities = set(stremio.detect_quality(s) for s in streams)
            log_service.error(
                f"Available qualities in addon response: {available_qualities}"
            )
            raise HTTPException(
                status_code=404, detail="No suitable stream quality found"
            )

        log_service.stream(
            f"Resolved {state_key} quality={quality} index={use_index} attempt={state.attempt_count} → {stream_url[:100]}..."
        )

        # Generate session ID and store stream URL for proxying
        session_id = generate_session_id(media_type, tmdb_id, quality, season, episode)
        _stream_sessions[session_id] = {
            "url": stream_url,
            "created": datetime.utcnow().timestamp()
        }
        
        # Cleanup old sessions occasionally
        if len(_stream_sessions) > 100:
            cleaned = cleanup_expired_sessions()
            if cleaned > 0:
                log_service.info(f"Cleaned up {cleaned} expired stream sessions")
        
        # Return proxy URL instead of direct redirect
        proxy_url = f"/api/stream/proxy/{session_id}"
        log_service.info(f"Created proxy session {session_id} -> {stream_url[:50]}...")
        
        return RedirectResponse(url=proxy_url, status_code=302)

    except HTTPException:
        raise
    except Exception as e:
        log_service.error(f"Stream resolution error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to resolve stream: {str(e)}"
        )
    finally:
        if tmdb:
            await tmdb.close()
        await stremio.close()


@router.get("/get-stream-url/{session_id}")
async def get_stream_url(session_id: str, secret: str = Query(...)):
    """
    Internal API to get stream URL by session ID.
    Used by stream server to fetch resolved stream URLs from main server.
    """
    if secret != _internal_api_secret:
        raise HTTPException(status_code=401, detail="Invalid secret")
    
    session_data = _stream_sessions.get(session_id)
    
    if not session_data:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "url": session_data["url"],
        "created": session_data["created"]
    }


@router.get("/proxy/{session_id}")
async def proxy_stream(session_id: str, request: Request):
    """
    Proxy endpoint that fetches the stream from debrid and forwards it to Jellyfin.
    This hides the debrid service URL from Jellyfin.
    Supports multiple concurrent streams.
    """
    log_service.info(f"[PROXY] Starting proxy for session {session_id}")
    
    # Check concurrent stream limit
    global _active_streams
    async with _stream_lock:
        if _active_streams >= MAX_CONCURRENT_STREAMS:
            log_service.error(f"Max concurrent streams ({MAX_CONCURRENT_STREAMS}) reached")
            raise HTTPException(
                status_code=503, 
                detail=f"Server busy - max {MAX_CONCURRENT_STREAMS} concurrent streams allowed"
            )
        _active_streams += 1
    
    # Try to get session from local storage first
    session_data = _stream_sessions.get(session_id)
    
    log_service.info(f"Looking up session {session_id}, found: {session_data is not None}")
    log_service.info(f"Available sessions: {list(_stream_sessions.keys())[:5]}...")  # Show first 5
    
    # If not found locally, try to fetch from main server (for stream server)
    jfresolve_url = getattr(settings, 'JFRESOLVE_SERVER_URL', None)
    if not session_data and jfresolve_url and isinstance(jfresolve_url, str):
        try:
            main_server_url = jfresolve_url.rstrip("/")
            api_url = f"{main_server_url}/api/stream/get-stream-url/{session_id}?secret={_internal_api_secret}"
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(api_url)
                if resp.status_code == 200:
                    session_data = resp.json()
                    log_service.info(f"Fetched stream URL from main server for session {session_id}")
                else:
                    log_service.error(f"Failed to fetch stream URL: {resp.status_code}")
        except Exception as e:
            log_service.error(f"Error fetching stream URL from main server: {e}")
    
    if not session_data:
        async with _stream_lock:
            _active_streams -= 1
        log_service.error(f"Invalid or expired session: {session_id}")
        raise HTTPException(status_code=404, detail="Session expired or invalid")
    
    stream_url = session_data.get("url")
    if not stream_url or not isinstance(stream_url, str):
        async with _stream_lock:
            _active_streams -= 1
        log_service.error(f"Invalid stream URL in session {session_id}: {type(stream_url).__name__}")
        raise HTTPException(status_code=502, detail="Invalid stream URL in session")
    
    stream_url = stream_url.strip()
    if not stream_url.lower().startswith(("http://", "https://")):
        async with _stream_lock:
            _active_streams -= 1
        log_service.error(f"Stream URL is not absolute: {stream_url[:80]}...")
        raise HTTPException(status_code=502, detail="Stream URL must be http(s)")
    
    log_service.info(f"Proxying stream for session {session_id} (active: {_active_streams})")
    log_service.stream(f"Proxying: {stream_url[:80]}...")
    
    try:
        # Forward headers (except host); ensure values are strings for httpx
        headers = {}
        for key, value in request.headers.items():
            if key.lower() == "host":
                continue
            if value is not None:
                headers[key] = value if isinstance(value, str) else str(value)
        
        log_service.info(f"Fetching stream from: {stream_url[:100]}...")
        
        # Use streaming HTTP client (httpx uses .stream(), not get(stream=True))
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(3600.0),  # 1 hour timeout for large files
            follow_redirects=True,
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10)
        ) as client:
            # Mutable container for headers (filled when generator starts streaming)
            meta = {"content_type": "application/octet-stream", "response_headers": {}}

            async def content_generator():
                stream_started = False
                try:
                    async with client.stream("GET", stream_url, headers=headers) as response:
                        response.raise_for_status()
                        meta["content_type"] = response.headers.get(
                            "content-type", "application/octet-stream"
                        )
                        meta["response_headers"] = {
                            k: v
                            for k, v in response.headers.items()
                            if k.lower() not in (
                                "content-length",
                                "transfer-encoding",
                                "connection",
                            )
                        }
                        content_length = response.headers.get("content-length")
                        log_service.info(
                            f"Proxy stream started: {meta['content_type']}, length: {content_length}"
                        )
                        async for chunk in response.aiter_bytes(chunk_size=8192):
                            stream_started = True
                            yield chunk
                finally:
                    if session_id in _stream_sessions:
                        del _stream_sessions[session_id]
                    if stream_started:
                        async with _stream_lock:
                            _active_streams -= 1
                        log_service.info(
                            f"Stream session {session_id} ended (active: {_active_streams})"
                        )

            # Advance generator once to open the stream and fill meta
            gen = content_generator()
            try:
                first_chunk = await gen.__anext__()
            except StopAsyncIteration:
                first_chunk = b""  # empty stream; meta already set by generator
                async with _stream_lock:
                    _active_streams -= 1

            async def full_gen():
                yield first_chunk
                async for chunk in gen:
                    yield chunk

            return StreamingResponse(
                full_gen(),
                media_type=meta["content_type"],
                headers=meta["response_headers"],
            )
            
    except httpx.HTTPError as e:
        async with _stream_lock:
            _active_streams -= 1
        msg = f"HTTP error proxying stream: {e}"
        log_service.error(msg)
        log_service.error(traceback.format_exc())
        print(f"[PROXY ERROR] {msg}\n{traceback.format_exc()}", file=sys.stderr, flush=True)
        raise HTTPException(status_code=502, detail=f"Failed to fetch stream: {str(e)}")
    except Exception as e:
        async with _stream_lock:
            _active_streams -= 1
        msg = f"Error proxying stream: {type(e).__name__}: {e}"
        log_service.error(msg)
        log_service.error(f"Stream URL (first 120 chars): {stream_url[:120] if stream_url else 'N/A'}")
        log_service.error(traceback.format_exc())
        print(f"[PROXY ERROR] {msg}\nStream URL: {stream_url[:120] if stream_url else 'N/A'}\n{traceback.format_exc()}", file=sys.stderr, flush=True)
        raise HTTPException(status_code=500, detail=f"Proxy error: {str(e)}")
