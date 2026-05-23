from fastapi import FastAPI



from backend.apis.health_routes import (
    router as health_router
)

from backend.apis.session_routes import (
    router as session_router
)

from backend.apis.interview_routes import (
    router as interview_router
)

from backend.apis.report_route import (
    router as report_router
)

# =========================================================
# FastAPI App
# =========================================================

app = FastAPI(

    title="AI Interview System",

    version="1.0.0"
)

# =========================================================
# Register Routers
# =========================================================

app.include_router(
    health_router
)

app.include_router(
    session_router
)

app.include_router(
    interview_router
)

app.include_router(
    report_router
)


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    import uvicorn
    import os
    BASE_URL = os.getenv("BASE_URL")
    PORT = int(os.getenv("PORT"))
    uvicorn.run("main:app", host=BASE_URL, port=PORT, reload=False)
