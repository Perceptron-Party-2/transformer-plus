from inference import inference 
import fastapi

app = fastapi.FastAPI()

# run with - uvicorn server:app --host 0.0.0.0 --port 8080 --reload
@app.on_event("startup")
async def startup_event():
  print("Starting Up")


@app.on_event("shutdown")
async def startup_event():
  print("Shutting down")


@app.get("/")
def on_root():
  return { "message": "Hello App" }


@app.post("/text_to_sql")# replace with challenge name
async def text_to_sql(request: fastapi.Request): # replace with challenge name

  # text is in text field?
  text = (await request.json())["text"]
  result = inference(text)
  print("text", text)
  print("result", result)
  return result
