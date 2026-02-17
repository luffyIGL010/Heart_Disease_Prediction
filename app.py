import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64
 

st.title("Heart Disease Predictor")
tab1,tab2,tab3=st.tabs(['Predict','Bulk Predict','Model Information'])

with tab1:
    import streamlit as st
import pickle
import numpy as np

# Tabs create karna
tab1, tab2, tab3 = st.tabs(['Predict', 'Bulk Predict', 'Model Information'])

with tab1:
    # Input fields create karna
    age = st.number_input("Age (years)", min_value=0, max_value=150)
    sex = st.selectbox("Sex", ["Male", "Female"])
    chest_pain = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", min_value=0, max_value=300)
    cholesterol = st.number_input("Serum Cholesterol (mm/dl)", min_value=0)
    fasting_bs = st.selectbox("Fasting Blood Sugar", ["<= 120 mg/dl", "> 120 mg/dl"])
    resting_ecg = st.selectbox("Resting ECG Results", ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
    max_hr = st.number_input("Maximum Heart Rate Achieved", min_value=60, max_value=202)
    exercise_angina = st.selectbox("Exercise-Induced Angina", ["Yes", "No"])
    oldpeak = st.number_input("Oldpeak (ST Depression)", min_value=0.0, max_value=10.0)
    st_slope = st.selectbox("Slope of Peak Exercise ST Segment", ["Upsloping", "Flat", "Downsloping"])

    # --- Convert categorical inputs to numeric (Logic section) ---
    
    # Sex conversion
    sex = 0 if sex == "Male" else 1
    
    # Chest Pain conversion
    chest_pain = ["Atypical Angina", "Non-Anginal Pain", "Asymptomatic", "Typical Angina"].index(chest_pain)
    
    # Fasting Blood Sugar conversion
    fasting_bs = 1 if fasting_bs == "> 120 mg/dl" else 0
    
    # Resting ECG conversion
    resting_ecg = ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"].index(resting_ecg)
    
    # Exercise Angina conversion
    exercise_angina = 1 if exercise_angina == "Yes" else 0
    
    # ST Slope conversion
    st_slope = ["Upsloping", "Flat", "Downsloping"].index(st_slope)



    # Create a DataFrame with user inputs
    input_data = pd.DataFrame({
       'Age': [age],
       'Sex': [sex],
       'ChestPainType': [chest_pain],
       'RestingBP': [resting_bp],
       'Cholesterol': [cholesterol],
       'FastingBS': [fasting_bs],
       'RestingECG': [resting_ecg],
       'MaxHR': [max_hr],
       'ExerciseAngina': [exercise_angina],
       'Oldpeak': [oldpeak],
       'ST_Slope': [st_slope]
    })

# Model mapping as per image
    algonames = ['Decision Trees', 'Logistic Regression', 'Random Forest', 'Support Vector Machine']
    modelnames = ['DecisionTree_model.pkl', 'LogisticR.pkl', 'RandomForest_model.pkl', 'SVM_model.pkl']


    predictions=[]

    def predict_heart_disease(data):
        for modelname in modelnames:
            model=pickle.load(open(modelname,'rb'))
            prediction=model.predict(data)
            predictions.append(prediction)

        return predictions


    if st.button('Submit'):
        st.subheader('Results.....')    
        st.markdown('-------------------------------')

        result=predict_heart_disease(input_data)

        for i in range(len(predictions)):
            st.subheader(algonames[i])
            if result[i][0]==0:
                st.write("No Heart disease detected.") 

            else:
                st.write("Heart disease detected.") 
            st.markdown('-----------------------')  


with tab2:
    st.title("Upload CSV File")
    
    st.subheader('Instructions to note before uploading the file:')
    st.info("""
    1. No NaN values allowed.
    2. Total 11 features in this order ('Age', 'Sex', 'ChestPainType', 'RestingBP', 'Cholesterol', 'FastingBS', 'RestingECG', 'MaxHR', 'ExerciseAngina', 'Oldpeak', 'ST_Slope').\n
    3. Check the spellings of the feature names.\n
    4. Feature values conventions:\n
       - Age: age of the patient [years]\n
       - Sex: [0: Male, 1: Female]\n
       - ChestPainType: [3: Typical Angina, 0: Atypical Angina, 1: Non-Anginal Pain, 2: Asymptomatic]\n
       - RestingBP: resting blood pressure [mm Hg]\n
       - Cholesterol: serum cholesterol [mm/dl]\n
       - FastingBS: [1: if FastingBS > 120 mg/dl, 0: otherwise]\n
       - RestingECG: [0: Normal, 1: ST-T wave abnormality, 2: LVH]\n
       - MaxHR: maximum heart rate achieved [60-202]\n
       - ExerciseAngina: [1: Yes, 0: No]\n
       - Oldpeak: numeric value measured in depression\n
       - ST_Slope: [0: upsloping, 1: flat, 2: downsloping]\n
    """)

    # File uploader
with tab2:
    st.title("Upload CSV File")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Load data
        data = pd.read_csv(uploaded_file)
        st.write("Uploaded Data Preview:")
        st.dataframe(data.head())

        # Model selection for bulk prediction
        selected_model_bulk = st.selectbox("Select Model for Bulk Prediction", algonames, key="bulk_key")
        model_file_bulk = modelnames[algonames.index(selected_model_bulk)]

        if st.button("Predict All"):
            try:
                # 1. Data ki copy banana taaki original file kharab na ho
                df_to_predict = data.copy()

                # 2. FIX: Agar 'HeartDisease' column hai, toh use hatao (ValueError Fix)
                if 'HeartDisease' in df_to_predict.columns:
                    df_to_predict = df_to_predict.drop(columns=['HeartDisease'])

                # 3. FIX: Agar data mein 'M/F' ya 'Normal/ST' jaise words hain, unhe numbers mein badlo
                # Ye mapping aapki images ke logic ke hisaab se hai
                if df_to_predict['Sex'].dtype == 'object':
                    df_to_predict['Sex'] = df_to_predict['Sex'].map({'M': 0, 'F': 1})
                
                if df_to_predict['ChestPainType'].dtype == 'object':
                    cp_map = {"ATA": 0, "NAP": 1, "ASY": 2, "TA": 3}
                    df_to_predict['ChestPainType'] = df_to_predict['ChestPainType'].map(cp_map)

                if df_to_predict['RestingECG'].dtype == 'object':
                    ecg_map = {"Normal": 0, "ST": 1, "LVH": 2}
                    df_to_predict['RestingECG'] = df_to_predict['RestingECG'].map(ecg_map)

                if df_to_predict['ExerciseAngina'].dtype == 'object':
                    df_to_predict['ExerciseAngina'] = df_to_predict['ExerciseAngina'].map({'N': 0, 'Y': 1})

                if df_to_predict['ST_Slope'].dtype == 'object':
                    slope_map = {"Up": 0, "Flat": 1, "Down": 2}
                    df_to_predict['ST_Slope'] = df_to_predict['ST_Slope'].map(slope_map)

                # 4. Model load aur predict
                with open(model_file_bulk, 'rb') as f:
                    loaded_model = pickle.load(f)
                
                predictions = loaded_model.predict(df_to_predict)
                
                # 5. Result dikhana
                data['Heart_Disease_Prediction'] = predictions
                data['Result'] = data['Heart_Disease_Prediction'].apply(lambda x: 'Disease' if x == 1 else 'Normal')
                
                st.success(f"Predictions completed using {selected_model_bulk}!")
                st.dataframe(data)
                
            except Exception as e:
                st.error(f"Error: {e}")





with tab3:
    st.header("📊 Model Performance & Technical Summary")
    st.write("Yahan aap training phase ke dauran mile models ke results aur unki details dekh sakte hain.")

    # Model Performance Table (Based on your training images)
    performance_data = {
        'Algorithm': ['Logistic Regression', 'Support Vector Machine', 'Decision Tree', 'Random Forest'],
        'Metric Used': ['Accuracy Score', 'F1-Score (Weighted)', 'Accuracy Score', 'Accuracy Score'],
        'Training Result': ['0.8641 (86.4%)', '0.8422 (84.2%)', '0.8097 (80.9%)', '0.8641 (86.4%)']
    }
    
    perf_df = pd.DataFrame(performance_data)
    st.table(perf_df)

    # --- Technical Details Section ---
    st.subheader("🛠 Technical Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Optimization Techniques:**")
        st.info("""
        - **GridSearchCV:** Decision Tree aur Random Forest ke best parameters dhoondne ke liye use kiya gaya.
        - **Cross-Validation:** Sabhi models ko 5-fold CV (cv=5) par test kiya gaya.
        - **Solvers/Kernels:** Logistic Regression mein multiple solvers aur SVM mein 'rbf', 'poly', 'linear' test kiye gaye.
        """)

    with col2:
        st.markdown("**Dataset Features:**")
        st.info("""
        - **Total Features:** 11 (Age, Sex, ChestPainType, etc.)
        - **Preprocessing:** Categorical data ko numeric indices mein badla gaya hai.
        - **Class Balance:** Decision Tree mein 'balanced' weight use kiya gaya hai.
        """)

    # Feature Importance Note
    st.warning("""
    **Important Note:** Model ki accuracy is baat par depend karti hai ki input data sahi format mein ho. 
    Bulk upload ke liye hamesha provided CSV format hi use karein.
    """)