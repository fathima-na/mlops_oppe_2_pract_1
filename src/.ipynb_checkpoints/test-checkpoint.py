import joblib
import pandas as pd
def predict_species(features: dict):
    model = joblib.load("model/model.pkl")
    input_df = pd.DataFrame([features])
    prediction = int(model.predict(input_df)[0])
    return {
        "predicted_class": prediction
}

print(predict_species({"alcohol":14.23,"malic_acid":1.71,"ash":2.43,"alcalinity_of_ash":15.6,"magnesium":127,"total_phenols":2.8,"flavanoids":3.06
,"nonflavanoid_phenols":0.28,"proanthocyanins":2.29,"color_intensity":5.64,"hue":1.04,"od280/od315_of_diluted_wines":3.92,"proline":1065}))