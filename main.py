"""
Experiment 6 -- FastAPI service for the BlinkIT Item_Outlet_Sales model
trained/explained in Experiments 4 and 5.
"""
from contextlib import asynccontextmanager

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = "blinkit_best_model.pkl"
ENCODERS_PATH = "blinkit_label_encoders.pkl"

# Column order the model was trained on (must match Experiment 5's X.columns)
# UPDATED: Added engineered features
FEATURE_ORDER = [
    "Item_Weight",
    "Item_Fat_Content",
    "Item_Visibility",
    "Item_Type",
    "Item_MRP",
    "Outlet_Establishment_Year",
    "Outlet_Size",
    "Outlet_Location_Type",
    "Outlet_Type",
    "Item_Type_Combined",
    "Outlet_Age",
    "Price_per_Weight",
    "Zero_Visibility_Flag",
]

ml_artifacts = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    ml_artifacts["model"] = joblib.load(MODEL_PATH)
    ml_artifacts["encoders"] = joblib.load(ENCODERS_PATH)
    print("Model and encoders loaded.")
    yield
    ml_artifacts.clear()


app = FastAPI(
    title="BlinkIT Sales Prediction API",
    description="Experiment 6: Containerization & API Deployment for the "
                 "BlinkIT Item_Outlet_Sales regression model.",
    version="1.0.0",
    lifespan=lifespan,
)


class BlinkITItem(BaseModel):
    Item_Weight: float = Field(..., example=9.3)
    Item_Fat_Content: str = Field(..., example="Low Fat")
    Item_Visibility: float = Field(..., example=0.016)
    Item_Type: str = Field(..., example="Dairy")
    Item_MRP: float = Field(..., example=249.81)
    Outlet_Establishment_Year: int = Field(..., example=1999)
    Outlet_Size: str = Field(..., example="Medium")
    Outlet_Location_Type: str = Field(..., example="Tier 1")
    Outlet_Type: str = Field(..., example="Supermarket Type1")


class PredictionResponse(BaseModel):
    predicted_item_outlet_sales: float


@app.get("/")
def health_check():
    return {"status": "ok", "message": "BlinkIT Sales Prediction API is running"}


@app.post("/predict", response_model=PredictionResponse)
def predict(item: BlinkITItem):
    if "model" not in ml_artifacts:
        raise HTTPException(status_code=503, detail="Model not loaded")

    model = ml_artifacts["model"]
    encoders = ml_artifacts["encoders"]

    # Convert input item to a DataFrame
    df = pd.DataFrame([item.model_dump()])

    # --- Feature Engineering (must match Experiment 4/5) ---
    # 1. Item_Type_Combined: Group Item_Type into broader categories
    def get_item_type_combined(item_type):
        if item_type in ['Soft Drinks', 'Hard Drinks']:
            return 'Drinks'
        elif item_type in ['Household', 'Health and Hygiene', 'Others']:
            return 'Non-Consumable'
        else:
            return 'Food' # Default for other food-related items

    df['Item_Type_Combined'] = df['Item_Type'].apply(get_item_type_combined)

    # 2. Outlet_Age: Calculate age based on a reference year (e.g., 2013 for this dataset)
    df['Outlet_Age'] = 2013 - df['Outlet_Establishment_Year']

    # 3. Price_per_Weight
    df['Price_per_Weight'] = df['Item_MRP'] / df['Item_Weight']

    # 4. Zero_Visibility_Flag
    df['Zero_Visibility_Flag'] = (df['Item_Visibility'] == 0).astype(int)
    # --- End Feature Engineering ---

    # Select and order features for the model
    row = df[FEATURE_ORDER]

    # Apply label encoders to categorical columns
    categorical_cols_to_encode = ['Item_Fat_Content', 'Item_Type', 'Outlet_Size',
                                  'Outlet_Location_Type', 'Outlet_Type', 'Item_Type_Combined']

    for col in categorical_cols_to_encode:
        if col in row.columns and col in encoders:
            try:
                row[col] = encoders[col].transform(row[col].astype(str))
            except ValueError as e:
                # This handles unseen categories for label encoded features
                raise HTTPException(
                    status_code=400,
                    detail=f"Unseen category '{row[col].iloc[0]}' for column '{col}': {e}",
                )

    prediction = model.predict(row)[0]
    return PredictionResponse(predicted_item_outlet_sales=round(float(prediction), 2))
