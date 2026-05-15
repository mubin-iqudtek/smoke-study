# -----------------------------------------------------------------------------
# VIDEO DOWNLOADER MODULE
# -----------------------------------------------------------------------------
# This file handles fetching videos from external sources.
# It supports:
# 1. YouTube URLs: Uses yt-dlp to download and merge video/audio into MP4.
# 2. Direct MP4 URLs: Uses requests to stream and save the video file.
# 3. Cache Management: Stores downloaded files in a temporary "cache/" folder.
# -----------------------------------------------------------------------------

import os
import uuid
import requests

CACHE_DIR = "cache"

os.makedirs(CACHE_DIR, exist_ok=True)


def is_youtube_url(url: str):

    return (
        "youtube.com" in url
        or "youtu.be" in url
    )


def download_youtube(url: str):

    import yt_dlp

    output_template = os.path.join(
        CACHE_DIR,
        f"{uuid.uuid4()}.%(ext)s"
    )

    ydl_opts = {

        # Better format handling
        "format": "bestvideo+bestaudio/best",

        # Output path
        "outtmpl": output_template,

        # Merge to mp4
        "merge_output_format": "mp4",

        # No playlist
        "noplaylist": True,

        # Reduce logs
        "quiet": False,

        # Ignore cert issues
        "nocheckcertificate": True,

        # Prevent IPv6 issues
        "source_address": "0.0.0.0",
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:

        info = ydl.extract_info(
            url,
            download=True
        )

        downloaded_file = ydl.prepare_filename(info)

        base, ext = os.path.splitext(downloaded_file)

        final_file = base + ".mp4"

        # Ensure file exists
        if not os.path.exists(final_file):

            # fallback
            final_file = downloaded_file

        return final_file


def download_direct_video(url: str):

    output_path = os.path.join(
        CACHE_DIR,
        f"{uuid.uuid4()}.mp4"
    )

    response = requests.get(
        url,
        stream=True,
        timeout=60
    )

    response.raise_for_status()

    with open(output_path, "wb") as f:

        for chunk in response.iter_content(8192):

            if chunk:
                f.write(chunk)

    return output_path


def download_video(url: str):

    if is_youtube_url(url):
        return download_youtube(url)

    return download_direct_video(url)