from fastapi import FastAPI
from pydantic import BaseModel
from src.models.predict import predict
import gradio as gr

app = FastAPI(
    title = "Marketing Target Identifier",
    description = "ML API for identifying customers who are likely to subscribe to a loan for a bank",
    version = "1.0.0"

)

@app.get("/")
async def root():
    """
    Health check
    """
    return {"status": "ok"}

class CustomerData(BaseModel):
    age: int
    job: str
    marital: str
    education: str
    default: str
    balance: int
    housing: str
    loan: str
    contact: str

class Prediction(BaseModel):
    prediction: str

@app.post("/predict", response_model=Prediction)
def get_prediction(data: CustomerData):
    try:
        result = predict(data.model_dump())
        return {"prediction": result}
    except Exception as e:
        return {"error": e}
    
def gradio_interface(
    age, job, marital, education, default,
    balance, housing,loan, contact
):
    data = {
        "age": float(age),
        "job": job,
        "marital": marital,
        "education": education,
        "default": default,
        "balance": float(balance),
        "housing": housing,
        "loan": loan,
        "contact": contact
    }
    result = predict(data)
    return str(result)

demo = gr.Interface(
    fn=gradio_interface,
    inputs=[
        gr.Number(label="age"),
        gr.Dropdown(["blue-collar", "management", "techinician", "services","admin.","business-owner","non-working","unknown"], label="job"),
        gr.Dropdown(["single", "married", "divorced"], label="marital"),
        gr.Dropdown(["primary", "secondary", "tertiary", "unknown"], label="education"),
        gr.Dropdown(["yes", "no"], label="default"),
        gr.Number(label="balance"),
        gr.Dropdown(["yes", "no"], label="housing"),
        gr.Dropdown(["yes", "no"], label="loan"),
        gr.Dropdown(["cellular", "telephone", "unknown"], label="contact"),
    ],
    outputs="text",
    title="Marketing Target Identifier",
    description="Fill in the customer details to determine if the person is likely to subscribe to a long-term loan.",
)

app = gr.mount_gradio_app(app, demo, path="/ui")