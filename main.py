import asyncio
from contextlib import asynccontextmanager
from functools import partial

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from scanner_service import VitalsScanner

_scan_lock = asyncio.Lock()
_DRAIN_SEC = 0.03


async def _drain_pending(ws: WebSocket, first_msg: dict):
    """
    Drop stale frames queued while the server was busy (slow VPS).
    Returns (latest_frame_or_none, priority_action_or_none).
    """
    latest_frame = first_msg if first_msg.get("type") == "frame" and first_msg.get("data") else None
    priority = None
    if first_msg.get("action") in ("start", "stop", "finish"):
        priority = first_msg

    while True:
        try:
            msg = await asyncio.wait_for(ws.receive_json(), timeout=_DRAIN_SEC)
        except asyncio.TimeoutError:
            break
        action = msg.get("action")
        if action in ("start", "stop", "finish"):
            priority = msg
            if action == "finish":
                break
        elif msg.get("type") == "frame" and msg.get("data"):
            latest_frame = msg
    return latest_frame, priority


@asynccontextmanager
async def lifespan(app: FastAPI):
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, VitalsScanner.preload_model)
    yield


app = FastAPI(title="rPPG Vitals API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "rPPG Vitals API", "model_loaded": True}


async def _handle_finish(loop, scanner, websocket):
    try:
        event = await loop.run_in_executor(None, scanner.finalize_scan)
    except Exception as exc:
        await websocket.send_json({"type": "error", "message": str(exc)})
        return
    await websocket.send_json(event)
    scanner.finish_scan()


@app.websocket("/ws/scan")
async def scan_websocket(websocket: WebSocket):
    await websocket.accept()

    async with _scan_lock:
        loop = asyncio.get_running_loop()
        scanner = VitalsScanner(duration=30, remote_mode=True)
        scanner.load_model()

        try:
            await websocket.send_json({
                "type": "status",
                "message": "Center your face in the camera",
                "phase": "preview",
            })

            while not scanner.stop_flag:
                msg = await websocket.receive_json()

                action = msg.get("action")
                if action == "start":
                    scanner.start_scan(int(msg.get("duration", 30)))
                    await websocket.send_json({
                        "type": "status",
                        "message": "Scanning…",
                        "phase": "scanning",
                    })
                    continue
                if action == "stop":
                    scanner.abort_scan()
                    await websocket.send_json({
                        "type": "status",
                        "message": "Stopped",
                        "phase": "preview",
                    })
                    continue
                if action == "finish":
                    await _handle_finish(loop, scanner, websocket)
                    continue

                if msg.get("type") != "frame":
                    continue

                latest_frame, priority = await _drain_pending(websocket, msg)

                if priority and priority.get("action") == "finish":
                    await _handle_finish(loop, scanner, websocket)
                    continue
                if priority and priority.get("action") == "start":
                    scanner.start_scan(int(priority.get("duration", 30)))
                    await websocket.send_json({
                        "type": "status",
                        "message": "Scanning…",
                        "phase": "scanning",
                    })
                    continue
                if priority and priority.get("action") == "stop":
                    scanner.abort_scan()
                    await websocket.send_json({
                        "type": "status",
                        "message": "Stopped",
                        "phase": "preview",
                    })
                    continue

                if not latest_frame:
                    continue

                try:
                    if latest_frame.get("ts") is not None:
                        scanner.note_client_timestamp(latest_frame["ts"])

                    client_face_ok = latest_frame.get("face_ok", True)
                    bgr = await loop.run_in_executor(
                        None, scanner.decode_frame, latest_frame["data"],
                    )
                    event = await loop.run_in_executor(
                        None, partial(scanner.process_frame, bgr, client_face_ok),
                    )
                except Exception as exc:
                    await websocket.send_json({"type": "error", "message": str(exc)})
                    continue

                if event:
                    await websocket.send_json(event)
                    if event.get("type") == "complete":
                        scanner.finish_scan()

        except WebSocketDisconnect:
            pass
        except Exception as exc:
            try:
                await websocket.send_json({"type": "error", "message": str(exc)})
            except WebSocketDisconnect:
                pass
        finally:
            scanner.stop()
