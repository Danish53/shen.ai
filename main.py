import asyncio
from contextlib import asynccontextmanager
from functools import partial

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from scanner_service import VitalsScanner

_scan_lock = asyncio.Lock()


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
                    try:
                        event = await loop.run_in_executor(None, scanner.finalize_scan)
                    except Exception as exc:
                        await websocket.send_json({"type": "error", "message": str(exc)})
                        continue
                    await websocket.send_json(event)
                    scanner.finish_scan()
                    continue

                if msg.get("type") != "frame":
                    continue

                jpeg_b64 = msg.get("data")
                if not jpeg_b64:
                    continue

                try:
                    if msg.get("ts") is not None:
                        scanner.note_client_timestamp(msg["ts"])

                    client_face_ok = msg.get("face_ok", True)
                    bgr = await loop.run_in_executor(None, scanner.decode_frame, jpeg_b64)
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
