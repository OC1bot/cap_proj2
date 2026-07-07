"""
app.py - Streamlit front end for the "What's In This Image?" project.

This file only handles the user interface: the sidebar for connection
settings, the image upload with validation, and the follow-up chat panel.
All AI logic (chains, prompts, parsing, memory) lives in pipeline.py.

Run with:
    streamlit run app.py

Important Streamlit concept used throughout this file: Streamlit reruns
this ENTIRE script from top to bottom on every user interaction (every
click, every keystroke committed). Ordinary variables are therefore lost
between interactions; anything that must survive is kept in
st.session_state, which persists for the whole browser session.
"""

import io                      # wraps raw bytes so Pillow can open them like a file
import time                    # timestamp used to throttle live-stream auto analysis

import streamlit as st         # the web UI framework
from PIL import Image          # used here only to validate and preview the upload

# Everything AI-related is imported from the pipeline module.
from pipeline import (
    DEFAULT_BASE_URL,
    analyze_image_with_diagnostics,  # analysis + failure case detection in one call          # http://localhost:11434, prefilled in Local mode
    analyze_video_with_diagnostics,  # NEW: frame sampling + per-frame analysis + aggregation
    DEFAULT_MODEL,             # preselected in the dropdown when available
    InMemoryChatMessageHistory,  # the LangChain memory object for follow-ups
    build_describe_chain,      # LCEL chain: preprocess | prompt | model | parser
    build_followup_chain,      # memory-backed chain for follow-up questions
    encode_image_bytes,        # bytes -> base64, cached for the no-re-upload feature
    frame_array_to_jpeg_bytes, # numpy BGR frame -> JPEG bytes, for the live stream branch
    check_ollama_vision,       # asks Ollama whether a model can accept images
    check_remote_vision,       # probes a remote endpoint with a 1x1 test image
    list_ollama_models,        # asks a local Ollama server what is installed
    list_openai_models,        # asks a remote OpenAI compatible endpoint the same
    make_llm,                  # builds a ChatOllama model (Local mode)
    make_remote_llm,           # builds a ChatOpenAI model (Remote mode)
)

# Browser tab title, icon, and a wide page so the two columns fit nicely.
st.set_page_config(page_title="What's In This Image?", page_icon="🖼", layout="wide")

# ---------------------------------------------------------------------------
# Cached model discovery
#
# Because Streamlit reruns the whole script constantly, calling the model
# listing endpoints directly would hit the server on every keystroke.
# st.cache_data stores the result keyed by the arguments (URL, key) and
# reuses it until the ttl (time to live, in seconds) expires or the cache
# is cleared by the Refresh button.
# ---------------------------------------------------------------------------

@st.cache_data(ttl=30, show_spinner=False)
def cached_local_models(url: str) -> list:
    """Cache the Ollama /api/tags call between Streamlit reruns."""
    return list_ollama_models(url)


@st.cache_data(ttl=60, show_spinner=False)
def cached_remote_models(url: str, key: str) -> list:
    """Cache the remote /models call between Streamlit reruns."""
    return list_openai_models(url, key)


@st.cache_data(ttl=300, show_spinner=False)
def cached_vision_check(model_name: str, url: str):
    """Cache the /api/show vision capability probe per model."""
    return check_ollama_vision(model_name, url)


@st.cache_data(ttl=300, show_spinner=False)
def cached_remote_vision_check(model_name: str, url: str, key: str):
    """Cache the remote 1x1 test image probe per model."""
    return check_remote_vision(model_name, url, key)


# Defaults for this rerun. They stay None until the user completes each
# sidebar step, and the code after the sidebar checks them to decide
# whether the app is ready to run.
model = None
mode = None
base_url = None
api_key = None
vision_blocked = False   # set True when the selected model fails the vision check

# ---------------------------------------------------------------------------
# Sidebar: connection settings, revealed step by step
#
# Step 1: choose Local or Remote.
# Step 2: enter the connection details for that choice.
#         (Remote also requires pressing Connect.)
# Step 3: pick a model from the list fetched from that server.
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Settings")

    # Step 1: pick the connection type. index=None means nothing is
    # preselected, so only the placeholder shows until the user chooses,
    # and the rest of the sidebar stays hidden.
    mode = st.selectbox(
        "Connection type",
        ["Local", "Remote"],
        index=None,
        placeholder="Select Local or Remote",
    )

    # Step 2 (Local): one field only, prefilled with the default address.
    if mode == "Local":
        base_url = st.text_input("Local Ollama URL", value=DEFAULT_BASE_URL)

    # Step 2 (Remote): URL + API key + Connect button.
    elif mode == "Remote":
        base_url = st.text_input(
            "Remote API URL",
            placeholder="https://api.openai.com/v1",
            help="Any OpenAI compatible endpoint: " 
                "OpenAI (https://api.openai.com/v1), "
                "Ollama Cloud (https://ollama.com/v1), "
                "OpenRouter (https://openrouter.ai/api/v1), " 
                "Hugging Face (https://router.huggingface.co/v1) and similar.",
        )
        # type="password" masks the key with dots as the user types.
        api_key = st.text_input("API key", type="password")

        # Security/consistency guard: if the user edits either field after
        # connecting, the old connection must not silently keep being used.
        # We remember the credentials that were connected with, and any
        # difference resets the connected flag, which brings the Connect
        # button back.
        creds = (base_url or "", api_key or "")
        if st.session_state.get("remote_creds") != creds:
            st.session_state.remote_connected = False

        # The Connect button is always visible in Remote mode. It stays
        # disabled (greyed out) until both fields are filled in, and it
        # disappears once connected.
        if not st.session_state.get("remote_connected"):
            connect_disabled = not (base_url and api_key)
            if st.button(
                "Connect",
                type="primary",              # orange primary styling
                use_container_width=True,    # full sidebar width
                disabled=connect_disabled,   # greyed out until fields are filled
            ):
                # Connecting = trying to list models. A wrong key or a dead
                # endpoint fails here, at Connect time, not later during
                # analysis, which gives the user immediate feedback.
                with st.spinner("Connecting..."):
                    found = cached_remote_models(base_url, api_key)
                if found:
                    st.session_state.remote_connected = True
                    st.session_state.remote_creds = creds
                    st.rerun()   # rerun immediately so the UI switches to the connected view
                else:
                    st.error(
                        f"Could not connect to {base_url}. Check the URL and "
                        "the API key, then press Connect again."
                    )
            if connect_disabled:
                st.caption("Enter the URL and API key to enable Connect.")

        # Green banner confirming which endpoint is connected.
        if st.session_state.get("remote_connected"):
            st.success(f"Connected to {base_url}")

    # Step 3 gate: the model list only loads when the connection is ready.
    # Local is ready as soon as a URL exists; Remote is ready only after a
    # successful Connect.
    if mode == "Local":
        details_ready = bool(base_url)
    elif mode == "Remote":
        details_ready = bool(st.session_state.get("remote_connected"))
    else:
        details_ready = False

    if details_ready:
        # Clearing both caches forces a fresh call to the server, e.g.
        # after pulling a new model with `ollama pull`.
        if st.button("Refresh model list"):
            cached_local_models.clear()
            cached_remote_models.clear()

        # Fetch the model names from whichever backend is active.
        if mode == "Local":
            available = cached_local_models(base_url)
        else:
            available = cached_remote_models(base_url, api_key)

        if not available:
            # Empty list means the server could not be reached or returned
            # nothing; tell the user what to check.
            st.error(
                f"Could not list models from {base_url}. Check that the "
                "server is reachable"
                + (" and the API key is valid" if mode == "Remote" else "")
                + ", then press Refresh model list."
            )
        else:
            # Preselect the project default model if the server has it.
            # Matching is by exact name first, then by the part before the
            # colon, so DEFAULT_MODEL = "llava" also matches "llava:latest".
            default_index = 0
            for i, name in enumerate(available):
                if name == DEFAULT_MODEL or name.split(":")[0] == DEFAULT_MODEL:
                    default_index = i
                    break
            model = st.selectbox("Model", available, index=default_index)
            st.caption(f"{len(available)} models found on the server.")

            # Vision validation happens HERE, at selection time, so a text
            # only model is caught before the user ever presses Analyze
            # Image, instead of failing later with a 400 multimodal error.
            if mode == "Local" and model:
                verdict = cached_vision_check(model, base_url)
                if verdict is False:
                    st.error(
                        f"'{model}' is a text only model and cannot analyse "
                        "images. Select a vision capable model such as "
                        "llava or llama3.2-vision."
                    )
                    vision_blocked = True
                    model = None
                elif verdict is True:
                    st.caption("Vision capability confirmed for this model.")
                else:
                    st.warning(
                        "Could not verify whether this model supports "
                        "images. If analysis fails with a multimodal "
                        "error, pick a vision capable model instead."
                    )
            elif mode == "Remote" and model:
                # Remote providers expose no capability metadata, so the
                # probe actually sends a 1x1 test image with max_tokens=1
                # to this model and checks whether it is accepted. This
                # may consume a token or two on the provider account.
                with st.spinner("Checking vision capability..."):
                    verdict = cached_remote_vision_check(model, base_url, api_key)
                if verdict is False:
                    st.error(
                        f"'{model}' rejected image input, so it cannot "
                        "analyse images. Select a vision capable model "
                        "such as gpt-4o."
                    )
                    vision_blocked = True
                    model = None
                elif verdict is True:
                    st.caption(
                        "Vision capability confirmed with a tiny test request."
                    )
                else:
                    st.warning(
                        "Could not verify whether this model supports "
                        "images (the test request was inconclusive). If "
                        "analysis fails with a multimodal error, pick a "
                        "vision capable model instead."
                    )

    # Wipes the analysis, the cached image, and the LangChain memory so
    # the user can start over without restarting the app.
    if st.button("Reset conversation"):
        for key in (
            "history", "result", "image_b64", "chat_log",
            "image_name", "failure",
            # Video-related state added by the video/webcam features:
            "video_name", "video_frames", "source_kind",
            # Live-stream-specific keys:
            "last_live_analysis_time", "live_stream_started",
        ):
            st.session_state.pop(key, None)
        st.rerun()

# ---------------------------------------------------------------------------
# Guidance screen
#
# Until a model is selected (model is still None), the main page shows a
# message telling the user which sidebar step is missing, then st.stop()
# ends the script so none of the app below renders half-configured.
# ---------------------------------------------------------------------------

if model is None:
    st.title("What's In This Image?")
    if mode is None:
        st.info("Start in the sidebar: select Local or Remote as the connection type.")
    elif mode == "Remote" and not (base_url and api_key):
        st.info("Enter the remote API URL and API key in the sidebar, then press Connect.")
    elif mode == "Remote" and not st.session_state.get("remote_connected"):
        st.info("Press Connect in the sidebar to load the model list.")
    elif vision_blocked:
        st.info(
            "The selected model does not support images. Pick a vision "
            "capable model in the sidebar to continue."
        )
    else:
        st.info("Complete the connection details in the sidebar to load the model list.")
    st.stop()

# ---------------------------------------------------------------------------
# Session state initialisation
#
# This is where the "follow-up questions without re-upload" requirement is
# satisfied. The base64 image and the LangChain message history both live
# in st.session_state, so they survive every Streamlit rerun for the whole
# browser session.
# ---------------------------------------------------------------------------

if "history" not in st.session_state:
    st.session_state.history = InMemoryChatMessageHistory()   # LangChain memory object
if "chat_log" not in st.session_state:
    st.session_state.chat_log = []   # (role, text) pairs used only to redraw the chat UI


def get_history(session_id: str) -> InMemoryChatMessageHistory:
    """
    RunnableWithMessageHistory calls this on every invocation to fetch the
    memory object for the given session id. This app has one browser
    session, so it always returns the single history from session state.
    """
    return st.session_state.history


# Build the chat model for whichever backend the user configured, then
# build both chains around it. The chains are identical for Local and
# Remote because both model objects speak the same LangChain interface.
if mode == "Local":
    llm = make_llm(model=model, base_url=base_url)
else:
    llm = make_remote_llm(model, base_url, api_key)
describe_chain = build_describe_chain(llm)
followup_chain = build_followup_chain(llm, get_history)

# ---------------------------------------------------------------------------
# Main layout: title, then two equal columns.
# Left column  = upload, validation, Analyze button, structured result.
# Right column = follow-up chat about the analysed image.
# ---------------------------------------------------------------------------

st.title("What's In This Image?")
st.write(
    "Upload an image or a short video, or take a photo with your webcam, "
    "get an AI description, then ask follow-up questions."
)

left, right = st.columns([1, 1])

# The two allowed lists used by the three validation layers below.
# ALLOWED_EXTENSIONS checks the file NAME; ALLOWED_FORMATS checks what the
# file CONTENT actually is once Pillow has decoded it.
ALLOWED_EXTENSIONS = ("jpg", "jpeg", "png", "webp", "avif")
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP", "AVIF"}

# Video file extensions Streamlit's uploader should accept. Actual decoding
# is done by opencv (which uses ffmpeg under the hood), and coverage varies
# by build; MP4/H.264 is the safest common denominator.
ALLOWED_VIDEO_EXTENSIONS = ("mp4", "mov", "avi", "webm", "mkv", "m4v")

INPUT_MODES = (
    "Upload image",
    "Upload video",
    "Take photo (webcam)",
    "Live webcam stream",
)

with left:
    # Input source selector. horizontal=True lays the radio buttons in a
    # single row so they take less vertical space than a stacked list.
    input_mode = st.radio(
        "Input source",
        INPUT_MODES,
        horizontal=True,
        key="input_mode_radio",
    )

    # Mode change detector: when the user switches input source, discard
    # every artefact of the previous analysis so a stale description,
    # frame grid, or cached image never leaks into a new run.
    if st.session_state.get("_input_mode") != input_mode:
        st.session_state._input_mode = input_mode
        for k in (
            "result", "failure", "image_b64",
            "image_name", "video_name",
            "video_frames", "source_kind",
            # Live-stream-specific keys added by the streaming branch:
            "last_live_analysis_time", "live_stream_started",
        ):
            st.session_state.pop(k, None)
        st.session_state.history = InMemoryChatMessageHistory()
        st.session_state.chat_log = []

    # -----------------------------------------------------------------
    # Branch 1: Upload image (the original path)
    # -----------------------------------------------------------------
    if input_mode == "Upload image":
        uploaded = st.file_uploader(
            "Upload an image",
            type=list(ALLOWED_EXTENSIONS),
            help="JPG and PNG are fully supported; WEBP and AVIF also work.",
        )

        if uploaded is not None:
            # Validation step 1: file extension must be in the allowed list.
            ext = uploaded.name.rsplit(".", 1)[-1].lower() if "." in uploaded.name else ""
            if ext not in ALLOWED_EXTENSIONS:
                st.error(
                    f"'{uploaded.name}' has an invalid file type. Please upload "
                    "a correct format: JPG, JPEG, PNG, WEBP or AVIF."
                )
                st.stop()

            # Validation step 2: file must actually decode as an image.
            raw = uploaded.getvalue()
            try:
                preview = Image.open(io.BytesIO(raw))
                preview.load()
            except Exception:
                st.error(
                    f"'{uploaded.name}' is not a valid image or cannot be "
                    "decoded. Please upload a correct format: JPG, JPEG, PNG, "
                    "WEBP or AVIF. If this is a real AVIF file, your Pillow "
                    "installation may lack the codec; try: pip install --upgrade pillow"
                )
                st.stop()

            # Validation step 3: real decoded format must be in the allowed set.
            detected = (preview.format or "").upper()
            if detected not in ALLOWED_FORMATS:
                st.error(
                    f"'{uploaded.name}' is actually a {detected or 'unknown'} "
                    "file, which is not supported. Please upload a correct "
                    "format: JPG, JPEG, PNG, WEBP or AVIF."
                )
                st.stop()

            st.image(preview, use_container_width=True)

            # New file selected: clear the previous analysis and memory.
            if st.session_state.get("image_name") != uploaded.name:
                st.session_state.image_name = uploaded.name
                for k in ("result", "failure", "image_b64", "video_frames"):
                    st.session_state.pop(k, None)
                st.session_state.history = InMemoryChatMessageHistory()
                st.session_state.chat_log = []

            if st.button("Analyze Image", type="primary", use_container_width=True):
                with st.spinner("Analysing image with the vision model..."):
                    try:
                        result, failure = analyze_image_with_diagnostics(llm, raw)
                    except Exception as exc:
                        st.error(
                            f"Could not reach the model: {exc}. "
                            "Check that Ollama is running and the model is pulled."
                        )
                        st.stop()

                st.session_state.result = result
                st.session_state.failure = failure
                st.session_state.image_b64 = encode_image_bytes(raw)
                st.session_state.source_kind = "image"

                st.session_state.history = InMemoryChatMessageHistory()
                st.session_state.history.add_user_message(
                    "Please describe the image I uploaded."
                )
                st.session_state.history.add_ai_message(result.description)
                st.session_state.chat_log = []

    # -----------------------------------------------------------------
    # Branch 2: Upload video (new)
    #
    # HONEST LIMIT: the pipeline samples N frames and analyses each as a
    # still image. Motion and cuts between sampled frames are invisible
    # to the model.
    # -----------------------------------------------------------------
    elif input_mode == "Upload video":
        st.caption(
            "Video is sampled into a few still frames, each analysed as a "
            "separate image. Motion between frames is not seen by the model."
        )
        uploaded_video = st.file_uploader(
            "Upload a video",
            type=list(ALLOWED_VIDEO_EXTENSIONS),
            help=(
                "MP4/H.264 is the safest choice. Other formats depend on the "
                "ffmpeg build shipped with your opencv-python install."
            ),
        )

        if uploaded_video is not None:
            raw_video = uploaded_video.getvalue()
            # Streamlit's native video player handles preview and playback.
            st.video(raw_video)

            # A slider lets the user trade speed against coverage: fewer
            # frames means fewer model calls (and a faster answer), more
            # frames means better temporal coverage.
            n_frames = st.slider(
                "Frames to sample",
                min_value=2,
                max_value=8,
                value=4,
                help=(
                    "Each frame costs one model call. On a small local model "
                    "such as moondream, 4 frames typically finishes in seconds; "
                    "on llava it may take a minute."
                ),
            )

            # New file selected: clear the previous analysis and memory.
            if st.session_state.get("video_name") != uploaded_video.name:
                st.session_state.video_name = uploaded_video.name
                for k in ("result", "failure", "image_b64", "video_frames"):
                    st.session_state.pop(k, None)
                st.session_state.history = InMemoryChatMessageHistory()
                st.session_state.chat_log = []

            if st.button("Analyze Video", type="primary", use_container_width=True):
                with st.spinner(
                    f"Sampling {n_frames} frames and analysing each..."
                ):
                    try:
                        result, failure, per_frame = analyze_video_with_diagnostics(
                            llm, raw_video, n_frames=n_frames
                        )
                    except ImportError as exc:
                        # opencv-python is not installed.
                        st.error(str(exc))
                        st.stop()
                    except Exception as exc:
                        st.error(
                            f"Could not analyse the video: {exc}. "
                            "Check that Ollama is running, the model is pulled, "
                            "and the video codec is supported by your opencv build."
                        )
                        st.stop()

                if not per_frame:
                    # extract_video_frames returned nothing; nothing more to do.
                    st.error(failure or "No frames could be extracted from the video.")
                    st.stop()

                st.session_state.result = result
                st.session_state.failure = failure
                st.session_state.video_frames = per_frame
                st.session_state.source_kind = "video"

                # For follow-up questions we pick the middle frame as the
                # representative image. Multi-image follow-up is not portable
                # across all backends, so this stays simple and honest.
                mid_idx = len(per_frame) // 2
                mid_frame_bytes = per_frame[mid_idx][0]
                st.session_state.image_b64 = encode_image_bytes(mid_frame_bytes)

                st.session_state.history = InMemoryChatMessageHistory()
                st.session_state.history.add_user_message(
                    "Please describe the video I uploaded, based on the sampled frames."
                )
                st.session_state.history.add_ai_message(result.description)
                st.session_state.chat_log = []

    # -----------------------------------------------------------------
    # Branch 3: Take photo (webcam) (new)
    #
    # st.camera_input returns a JPEG snapshot when the user clicks the
    # capture button, so downstream it is identical to an image upload.
    # -----------------------------------------------------------------
    elif input_mode == "Take photo (webcam)":
        st.caption(
            "Your browser will ask for camera permission. Click the capture "
            "button, then Analyze Image."
        )
        captured = st.camera_input("Take a photo")

        if captured is not None:
            raw = captured.getvalue()
            # camera_input always returns a JPEG, so a lightweight decode
            # check is enough. The picker cannot supply anything else here.
            try:
                preview = Image.open(io.BytesIO(raw))
                preview.load()
            except Exception:
                st.error(
                    "The captured frame could not be decoded. Please try "
                    "capturing again."
                )
                st.stop()

            # camera_input does not give the capture a stable filename, so
            # we hash the bytes to detect "a new capture" vs "same one on
            # this rerun". A short hex tag is enough for the equality check.
            capture_tag = f"webcam_{abs(hash(raw)) & 0xffffff:06x}.jpg"
            if st.session_state.get("image_name") != capture_tag:
                st.session_state.image_name = capture_tag
                for k in ("result", "failure", "image_b64", "video_frames"):
                    st.session_state.pop(k, None)
                st.session_state.history = InMemoryChatMessageHistory()
                st.session_state.chat_log = []

            if st.button("Analyze Image", type="primary", use_container_width=True):
                with st.spinner("Analysing captured photo with the vision model..."):
                    try:
                        result, failure = analyze_image_with_diagnostics(llm, raw)
                    except Exception as exc:
                        st.error(
                            f"Could not reach the model: {exc}. "
                            "Check that Ollama is running and the model is pulled."
                        )
                        st.stop()

                st.session_state.result = result
                st.session_state.failure = failure
                st.session_state.image_b64 = encode_image_bytes(raw)
                st.session_state.source_kind = "webcam"

                st.session_state.history = InMemoryChatMessageHistory()
                st.session_state.history.add_user_message(
                    "Please describe the photo I just captured."
                )
                st.session_state.history.add_ai_message(result.description)
                st.session_state.chat_log = []

    # -----------------------------------------------------------------
    # Branch 4: Live webcam stream (new)
    #
    # HONEST LIMIT: Ollama vision models take seconds per frame, so this
    # cannot be true real-time analysis. What we build is a live video
    # preview via WebRTC, plus periodic snapshots that the model analyses
    # on a timer or on demand. Motion between snapshots is not seen.
    # -----------------------------------------------------------------
    elif input_mode == "Live webcam stream":
        st.caption(
            "Live browser video via WebRTC. Analysis runs on a timer or on "
            "the Capture button, not per frame, because each model call "
            "takes seconds."
        )

        # Optional dependencies. Import here so the rest of the app runs
        # for users who have not installed the streaming stack yet.
        try:
            from streamlit_webrtc import webrtc_streamer, WebRtcMode
        except ImportError:
            st.error(
                "Live streaming needs streamlit-webrtc. Install with:\n\n"
                "    pip install streamlit-webrtc\n\n"
                "It brings in aiortc and av as transitive dependencies; "
                "the whole install is around 50 to 100 MB."
            )
            st.stop()

        import threading  # standard library; local import keeps the top clean

        # Video processor: called by WebRTC on every incoming frame in a
        # worker thread. It only stashes the latest frame under a lock so
        # the main Streamlit thread can pick it up later. The frame is
        # returned unchanged so the browser keeps showing live video.
        # DO NOT touch st.session_state from recv(); Streamlit's session
        # state is not thread safe.
        class LiveFrameGrabber:
            def __init__(self):
                self.latest_frame = None
                self.lock = threading.Lock()

            def recv(self, frame):
                img = frame.to_ndarray(format="bgr24")
                with self.lock:
                    # copy() because the underlying buffer can be recycled
                    self.latest_frame = img.copy()
                return frame

        # SENDRECV keeps the video preview on screen; audio is off because
        # the vision model does not use it and disabling it avoids a
        # microphone permission prompt.
        ctx = webrtc_streamer(
            key="live-webcam-stream",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=LiveFrameGrabber,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

        # Controls row: auto analyse toggle + interval slider + manual button.
        col_a, col_b = st.columns([1, 2])
        with col_a:
            auto_analyze = st.checkbox("Auto analyse", value=False)
        with col_b:
            interval_seconds = st.slider(
                "Interval (seconds)",
                min_value=3,
                max_value=30,
                value=8,
                disabled=not auto_analyze,
                help=(
                    "How often to grab a snapshot when Auto analyse is on. "
                    "Model calls that take longer than the interval do not "
                    "queue up; the next snapshot is skipped instead."
                ),
            )

        stream_playing = bool(getattr(ctx, "state", None) and ctx.state.playing)
        processor_ready = getattr(ctx, "video_processor", None) is not None

        manual_click = st.button(
            "Capture and analyse now",
            type="primary",
            use_container_width=True,
            disabled=not (stream_playing and processor_ready),
        )

        # Wire the timer. streamlit-autorefresh triggers a rerun every
        # interval_seconds while Auto analyse is on. If it is not installed
        # we drop back to manual-only mode with a clear message.
        if auto_analyze and stream_playing:
            try:
                from streamlit_autorefresh import st_autorefresh
                st_autorefresh(
                    interval=interval_seconds * 1000,
                    limit=None,
                    key="live-stream-autorefresh",
                )
            except ImportError:
                st.warning(
                    "Auto analyse needs streamlit-autorefresh. Install with:\n\n"
                    "    pip install streamlit-autorefresh\n\n"
                    "Falling back to manual capture only."
                )
                auto_analyze = False

        # Decide whether THIS rerun should trigger a model call. The
        # timer-driven path is throttled by wall clock so an unusually
        # slow model call cannot cause pile-up.
        now = time.time()
        should_analyze = False
        if manual_click:
            should_analyze = True
        elif auto_analyze and stream_playing and processor_ready:
            last_ts = st.session_state.get("last_live_analysis_time", 0.0)
            if now - last_ts >= interval_seconds:
                should_analyze = True

        if should_analyze and processor_ready:
            # Copy the latest frame out from under the lock so the model
            # call below does not hold the lock for seconds and block the
            # WebRTC thread from writing new frames.
            with ctx.video_processor.lock:
                frame = (
                    None
                    if ctx.video_processor.latest_frame is None
                    else ctx.video_processor.latest_frame.copy()
                )

            if frame is None:
                st.info(
                    "Waiting for the first frame from the webcam..."
                )
            else:
                with st.spinner("Analysing latest frame..."):
                    try:
                        jpeg_bytes = frame_array_to_jpeg_bytes(frame)
                        result, failure = analyze_image_with_diagnostics(
                            llm, jpeg_bytes
                        )
                    except Exception as exc:
                        st.error(
                            f"Analysis failed: {exc}. Check that Ollama is "
                            "running and the model is pulled."
                        )
                        result, failure = None, None

                if result is not None:
                    st.session_state.result = result
                    st.session_state.failure = failure
                    st.session_state.image_b64 = encode_image_bytes(jpeg_bytes)
                    st.session_state.source_kind = "live"
                    st.session_state.last_live_analysis_time = now

                    # Seed the chat history once per streaming session, so
                    # follow-ups have a starting point. Subsequent
                    # analyses update image_b64 (the follow-up chat always
                    # sees the most recent frame) but do not re-seed the
                    # history, otherwise every timer tick would wipe the
                    # conversation.
                    if not st.session_state.get("live_stream_started"):
                        st.session_state.live_stream_started = True
                        st.session_state.history = InMemoryChatMessageHistory()
                        st.session_state.history.add_user_message(
                            "Please describe the current view from my live webcam."
                        )
                        st.session_state.history.add_ai_message(result.description)
                        st.session_state.chat_log = []

    # -----------------------------------------------------------------
    # Shared result display (works for all four modes)
    # -----------------------------------------------------------------
    if "result" in st.session_state:
        r = st.session_state.result
        st.markdown(f"**Description:** {r.description}")
        st.markdown(f"**Objects:** {', '.join(r.objects) if r.objects else 'none detected'}")
        st.markdown(f"**Scene type:** {r.scene_type}")
        if getattr(r, "limitations", None):
            st.markdown(f"**Limitations:** {', '.join(r.limitations)}")

        # Failure case notice: unchanged behaviour, still shown when the
        # model's reply deviated from the instructed JSON format.
        if st.session_state.get("failure"):
            st.warning(
                f"Failure case detected: {st.session_state.failure} "
                "You can use this run as the failure example in your write-up."
            )

        # For video runs, also show each sampled frame with its own
        # description. This makes the "N still images stitched together"
        # nature of video analysis honest and visible to the user.
        if st.session_state.get("source_kind") == "video" and st.session_state.get("video_frames"):
            st.markdown("---")
            st.markdown("**Sampled frames**")
            frames = st.session_state.video_frames
            for idx, (frame_bytes, frame_result, _) in enumerate(frames, start=1):
                st.image(frame_bytes, caption=f"Frame {idx}", use_container_width=True)
                st.caption(frame_result.description or "(no description)")

with right:
    st.subheader("Ask follow-up questions")

    # No analysed image yet: show guidance instead of an empty chat box.
    if "image_b64" not in st.session_state:
        st.info(
            "Choose an input source on the left, upload or capture, then "
            "press Analyze first."
        )
    else:
        # Source-specific honesty notes: what the follow-up chat can and
        # cannot see. Both cases share the same underlying limit: the
        # follow-up chain only ever has one image attached at a time.
        if st.session_state.get("source_kind") == "video":
            st.caption(
                "Follow-up questions are answered against the middle sampled "
                "frame only, not the whole video. For questions about other "
                "moments, mention the frame number in your question."
            )
        elif st.session_state.get("source_kind") == "live":
            st.caption(
                "Follow-up questions are answered against the most recently "
                "analysed frame from the live stream, not real time. If the "
                "scene has changed since the last analysis, capture a fresh "
                "one before asking."
            )

        # Redraw all previous turns. Necessary because Streamlit reruns
        # the script on every interaction and would otherwise show an
        # empty chat each time.
        for role, text in st.session_state.chat_log:
            with st.chat_message(role):
                st.write(text)

        # chat_input renders the text box pinned at the bottom and returns
        # the typed question once, on the rerun where the user submits it.
        question = st.chat_input("e.g. Is there a person in this image?")
        if question:
            # Show the user's bubble immediately.
            st.session_state.chat_log.append(("user", question))
            with st.chat_message("user"):
                st.write(question)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        # The memory-backed chain: past turns are injected
                        # automatically, the cached image is re-attached,
                        # and this new question + answer are appended to
                        # the history afterwards. session_id tells
                        # get_history() which conversation this belongs to.
                        answer = followup_chain.invoke(
                            {
                                "question": question,
                                "image_b64": st.session_state.image_b64,
                            },
                            config={"configurable": {"session_id": "streamlit"}},
                        )
                    except Exception as exc:
                        # A model error becomes a chat message rather than
                        # a crash, so the conversation can continue.
                        answer = f"Error talking to the model: {exc}"
                st.write(answer)

            # Store the answer so it is redrawn on future reruns.
            st.session_state.chat_log.append(("assistant", answer))