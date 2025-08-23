from fastapi import FastAPI, Request, HTTPException, Response, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError
import logging
import time
import json
import joblib
import sys
import pandas as pd
import time
import os
# OpenTelemetry imports
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter
from opentelemetry.trace import StatusCode

# Setup Tracer
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)
span_processor = BatchSpanProcessor(CloudTraceSpanExporter())
trace.get_tracer_provider().add_span_processor(span_processor)

# ---- App state flags ----
app_state = {"is_ready": False, "is_alive": True}

app = FastAPI(title="Classifier API")

# Load model
model_path = "model/model.pkl"
model = None

logger = logging.getLogger("api")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(message)s') 
handler.setFormatter(formatter)
logger.addHandler(handler)

class WineFeatures(BaseModel):
    alcohol: float; malic_acid: float; ash: float; alcalinity_of_ash: float
    magnesium: float; total_phenols: float; flavanoids: float; nonflavanoid_phenols: float
    proanthocyanins: float; color_intensity: float; hue: float
    od280_315_of_diluted_wines: float; proline: float

@app.on_event("startup")
async def startup_event():
    global model
    # CHeck path exists
    try: 
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            app_state["is_ready"] = True
            logger.info(json.dumps({
                    "event": "Model_load",
                    "status": "success"
                }))
        else:
            app_state["is_ready"] = False
            logger.error(json.dumps({
                    "event": "Model_load",
                    "error": "Model path does not exist: " + model_path
                }))
    except Exception as e:
        app_state["is_ready"] = False
        logger.error(json.dumps({
                "event": "Model_load",
                "error": str(e)
            }))

# Health probes
@app.get("/live_check", tags=["Probe"])
async def liveness_probe():
    if app_state["is_alive"]:
        return {"status": "alive"}
    return Response(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)

@app.get("/ready_check", tags=["Probe"])
async def readiness_probe():
    if app_state["is_ready"]:
        return {"status": "ready"}
    return Response(status_code=status.HTTP_503_SERVICE_UNAVAILABLE)

@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = round((time.time() - start_time) * 1000, 2)
    response.headers["X-Process-Time-ms"] = str(duration)
    return response

@app.exception_handler(Exception)
async def exception_handler(request: Request, exc: Exception):
    span = trace.get_current_span()
    trace_id = format(span.get_span_context().trace_id, "032x")
    logger.exception(json.dumps({
        "event": "unhandled_exception",
        "trace_id": trace_id,
        "path": str(request.url),
        "error": str(exc)
    }))
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error", "trace_id": trace_id},
    )

@app.get("/")
def read_root():
    return {"message": "Welcome to the Classifier API!"}

@app.post("/predict")
def predict_species(input: WineFeatures, request:Request):
    with tracer.start_as_current_span("model_inference") as span:
        if not app_state["is_ready"] or model is None:
            raise HTTPException(status_code=503, detail="App State/Model is not ready")
        
        start_time = time.time()
        trace_id = format(span.get_span_context().trace_id, "032x")
        
        FEATURE_ORDER = [
            "alcohol",
            "malic_acid",
            "ash",
            "alcalinity_of_ash",
            "magnesium",
            "total_phenols",
            "flavanoids",
            "nonflavanoid_phenols",
            "proanthocyanins",
            "color_intensity",
            "hue",
            "od280/od315_of_diluted_wines",   # keep original sklearn name
            "proline"
        ]

        try: 
            data = input.dict()
            data["od280/od315_of_diluted_wines"] = data.pop("od280_315_of_diluted_wines")
            o_data = {feat: data[feat] for feat in FEATURE_ORDER}
            input_df = pd.DataFrame([o_data])
            prediction = int(model.predict(input_df)[0])
            latency = round((time.time() - start_time) * 1000, 2)

            logger.info(json.dumps({
                "event": "prediction",
                "trace_id": trace_id,
                 "input": input_df.to_dict(orient="records")[0],
                "result": prediction,
                "latency_ms": latency,
                "status": "success"
            }))
            span.set_status(StatusCode.OK) 
            return {"predicted_class": prediction, "trace_id": trace_id}
        except Exception as e:
            logger.error({
                "event": "prediction_error",
                "error": str(e),
                "trace_id": trace_id
            })
            span.set_status(StatusCode.ERROR, description=str(e))
            raise HTTPException(status_code=500, detail="Prediction failed")