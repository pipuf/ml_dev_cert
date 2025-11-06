from fastapi import APIRouter
from pydantic import BaseModel, Field

from helpers.database import insert_prediction, get_predictions as get_db_predictions
from helpers.database import get_db
from helpers.model import predict as model_predict


router = APIRouter()


# Pydantic model for validating the input data
class Sample(BaseModel):
    gender: str = Field(..., pattern="^(Female|Male|Other)$")
    age: int = Field(..., ge=0)
    hypertension: int = Field(..., ge=0, le=1)
    heart_disease: int = Field(..., ge=0, le=1)
    smoking_history: str = Field(
        ..., pattern="^(never|current|former|ever|not current)$"
    )
    bmi: float = Field(..., ge=0)
    HbA1c_level: float = Field(..., ge=0)
    blood_glucose_level: int = Field(..., ge=0)



@router.post("/predict")
async def predict(sample: Sample):

    # Run model and get prediction
    prediction = model_predict(**sample.model_dump())

    # Insert prediction into the database
    insert_prediction(**sample.model_dump(), prediction=prediction)

    return {"prediction": prediction}


@router.get("/predictions")
async def get_predictions():
    return get_db_predictions()


@router.put("/predictions/{prediction_id}")
async def update_prediction(prediction_id: int, sample: Sample):
    """
    Update an existing prediction by ID.
    """
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        f"""
        UPDATE predictions
        SET 
            gender = "{sample.gender}",
            age = {sample.age},
            hypertension = {sample.hypertension},
            heart_disease = {sample.heart_disease},
            smoking_history = "{sample.smoking_history}",
            bmi = {sample.bmi},
            HbA1c_level = {sample.HbA1c_level},
            blood_glucose_level = {sample.blood_glucose_level},
            prediction = {model_predict(**sample.model_dump())}
        WHERE rowid = {prediction_id};
        """
    )
    conn.commit()
    cur.close()
    conn.close()
    return {"message": f"Prediction with ID {prediction_id} updated successfully."}


@router.delete("/predictions/{prediction_id}")
async def delete_prediction(prediction_id: int):
    """
    Delete a prediction by ID.
    """
    conn = get_db()
    cur = conn.cursor()
    cur.execute(f"DELETE FROM predictions WHERE rowid = {prediction_id};")
    conn.commit()
    cur.close()
    conn.close()
    return {"message": f"Prediction with ID {prediction_id} deleted successfully."}
