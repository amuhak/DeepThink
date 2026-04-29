import subprocess
import time
import sys

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Code Sandbox")

DEFAULT_TIMEOUT = 30


class CodeRequest(BaseModel):
    code: str
    timeout: int = DEFAULT_TIMEOUT


class CodeResponse(BaseModel):
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool


@app.post("/execute")
async def execute_code(req: CodeRequest):
    if req.timeout > 120:
        raise HTTPException(status_code=400, detail="Timeout cannot exceed 120 seconds")

    print(f"\n{'='*60}")
    print(f"[SANDBOX] Executing code (timeout={req.timeout}s):")
    print(f"{'-'*60}")
    print(req.code)
    print(f"{'-'*60}")

    try:
        proc = subprocess.run(
            [sys.executable, "-c", req.code],
            capture_output=True,
            text=True,
            timeout=req.timeout,
            env={
                "PYTHONIOENCODING": "utf-8",
                "PYTHONUNBUFFERED": "1",
                "PATH": "/usr/local/bin:/usr/bin:/bin",
                "HOME": "/tmp",
                "TMPDIR": "/tmp",
            },
        )

        print(f"[SANDBOX] Result: exit_code={proc.returncode}")
        if proc.stdout:
            print(f"[SANDBOX] stdout:\n{proc.stdout}")
        if proc.stderr:
            print(f"[SANDBOX] stderr:\n{proc.stderr}")
        print(f"{'='*60}\n")

        return CodeResponse(
            stdout=proc.stdout,
            stderr=proc.stderr,
            exit_code=proc.returncode,
            timed_out=False,
        )
    except subprocess.TimeoutExpired:
        print(f"[SANDBOX] TIMED OUT after {req.timeout}s")
        print(f"{'='*60}\n")
        return CodeResponse(
            stdout="",
            stderr="Execution timed out",
            exit_code=-1,
            timed_out=True,
        )
    except Exception as e:
        print(f"[SANDBOX] ERROR: {str(e)}")
        print(f"{'='*60}\n")
        return CodeResponse(
            stdout="",
            stderr=f"Sandbox error: {str(e)}",
            exit_code=-2,
            timed_out=False,
        )


@app.get("/health")
async def health():
    return {"status": "ok"}
