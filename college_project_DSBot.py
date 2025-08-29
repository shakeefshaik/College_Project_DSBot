import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from pycaret.classification import (
    setup as cls_setup,
    compare_models as cls_compare,
    finalize_model as cls_finalize,
    predict_model as cls_predict,
)
from pycaret.regression import (
    setup as reg_setup,
    compare_models as reg_compare,
    finalize_model as reg_finalize,
    predict_model as reg_predict,
)
from imblearn.over_sampling import SMOTE
import pickle
import spacy
import re
# Load SpaCy NLP model
nlp = spacy.load("en_core_web_sm")
class ExploratoryDataAnalysis:
    """Handles all EDA-related operations."""

    def display_head(self):
        df = st.session_state.cleaned_df
        if st.button("Show First 5 Rows"):
            st.subheader("📜 First 5 Rows of Dataset")
            st.dataframe(df.head())

    def show_shape(self):
        df = st.session_state.cleaned_df
        if st.button("Show Dataset Shape"):
            st.subheader("📏 Dataset Shape")
            st.write(df.shape)

    def show_columns(self):
        df = st.session_state.cleaned_df
        if st.button("Show Column Names"):
            st.subheader("🗂 Column Names")
            st.write(df.columns.tolist())

    def summary_statistics(self):
        df = st.session_state.cleaned_df
        if st.button("Show Summary Statistics"):
            st.subheader("📊 Summary Statistics")
            st.write(df.describe())

    def missing_values(self):
        df = st.session_state.cleaned_df
        if st.button("Check Missing Values"):
            st.subheader("🔍 Missing Value Analysis")
            st.write(df.isnull().sum())

    def correlation_analysis(self):
        df = st.session_state.cleaned_df
        st.subheader("📊 Correlation Matrix")
        selected_columns = st.multiselect("Select Columns for Correlation Analysis", df.columns.tolist())
        if st.button("Generate Correlation Matrix"):
            if selected_columns:
                plt.figure(figsize=(10, 6))
                sns.heatmap(df[selected_columns].corr(), annot=True, cmap="coolwarm")
                st.pyplot()
            else:
                st.warning("⚠ Please select at least one column for correlation analysis.")

    def feature_distribution(self):
        df = st.session_state.cleaned_df
        st.subheader("📈 Feature Distributions")
        num_cols = df.select_dtypes(include=np.number).columns
        selected_columns = st.multiselect("Select Columns for Distribution", num_cols)
        if st.button("Generate Distributions"):
            if selected_columns:
                for col in selected_columns:
                    plt.figure(figsize=(6, 4))
                    sns.histplot(df[col], kde=True)
                    st.pyplot()
            else:
                st.warning("⚠ Please select at least one column for feature distribution.")

    def outlier_detection(self):
        df = st.session_state.cleaned_df
        st.subheader("⚠ Outlier Detection Using Box Plots")
        num_cols = df.select_dtypes(include=np.number).columns
        selected_columns = st.multiselect("Select Columns for Outlier Detection", num_cols)
        if st.button("Generate Outlier Detection"):
            if selected_columns:
                for col in selected_columns:
                    plt.figure(figsize=(6, 4))
                    sns.boxplot(x=df[col])
                    st.pyplot()
            else:
                st.warning("⚠ Please select at least one column for outlier detection.")
class Cleaning:
    """Handles core data cleaning steps."""

    def remove_duplicates(self):
        df = st.session_state.cleaned_df
        st.header("1️⃣ Remove Duplicate Rows")
        if st.button("Remove Duplicates"):
            before = df.shape[0]
            df = df.drop_duplicates()
            after = df.shape[0]
            st.session_state.cleaned_df = df
            st.success(f"✅ Removed {before - after} duplicate rows.")
        st.dataframe(df)

    def drop_columns(self):
        df = st.session_state.cleaned_df
        st.header("2️⃣ Drop Columns")
        cols = st.multiselect("Select columns to remove", df.columns.tolist(), key="drop_cols")
        if st.button("Drop Selected Columns"):
            df = df.drop(columns=cols)
            st.session_state.cleaned_df = df
            st.success(f"✅ Dropped: {', '.join(cols)}")
        st.dataframe(df)

    def handle_missing(self):
        df = st.session_state.cleaned_df
        st.header("3️⃣ Handle Missing Values")
        missing_cols = [col for col in df.columns if df[col].isnull().sum() > 0]
        if not missing_cols:
            st.info("✅ No missing values.")
            return

        for col in missing_cols:
            st.write(f"📌 `{col}` — {df[col].isnull().sum()} missing")
            strat = st.selectbox(f"Strategy for `{col}`", ["Skip", "Drop Rows", "Mean", "Median", "Mode"], key=f"miss_{col}")
            if strat != "Skip" and st.button(f"Apply {strat} to `{col}`", key=f"btn_{col}"):
                if strat == "Drop Rows":
                    df = df[df[col].notnull()]
                elif strat == "Mean":
                    df[col] = df[col].fillna(df[col].mean())
                elif strat == "Median":
                    df[col] = df[col].fillna(df[col].median())
                elif strat == "Mode":
                    df[col] = df[col].fillna(df[col].mode().iloc[0])
                st.session_state.cleaned_df = df
                st.success(f"✅ Cleaned `{col}` with {strat}")
                st.dataframe(df)

    def scale_columns(self):
        df = st.session_state.cleaned_df
        st.header("4️⃣ Scale Numeric Columns")
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        selected = st.multiselect("Select numeric columns to scale", num_cols, key="scale_cols")
        method = st.radio("Scaling Method", ["Standard", "Min-Max"], key="scale_method")
        if st.button("Apply Scaling"):
            if selected:
                scaler = StandardScaler() if method == "Standard" else MinMaxScaler()
                df[selected] = scaler.fit_transform(df[selected])
                st.session_state.cleaned_df = df
                st.success(f"✅ Scaled {', '.join(selected)} using {method} scaler")
                st.dataframe(df)
            else:
                st.warning("⚠ Please select at least one column.")

    def encode_categoricals(self):
        df = st.session_state.cleaned_df
        st.header("5️⃣ Encode Categorical Columns")
        cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        selected = st.multiselect("Select columns to encode", cat_cols, key="encode_cols")
        if st.button("Apply Label Encoding"):
            encoder = LabelEncoder()
            for col in selected:
                df[col] = encoder.fit_transform(df[col].astype(str))
            st.session_state.cleaned_df = df
            st.success(f"✅ Encoded: {', '.join(selected)}")
            st.dataframe(df)

    def download_data(self):
        df = st.session_state.cleaned_df
        st.header("📥 Download Cleaned Data")
        csv = df.to_csv(index=False)
        st.download_button("Download CSV", data=csv, file_name="cleaned_dataset.csv", mime="text/csv")

    def reset(self):
        if st.button("🔄 Reset App"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.experimental_rerun()
class DataVisualization:
    """Handles custom plotting."""

    def generate_plot(self):
        df = st.session_state.cleaned_df
        st.subheader("📊 Generate Custom Plots")

        plot_type = st.selectbox("Select Plot Type", ["Line", "Bar", "Scatter", "Box"])
        x_axis = st.selectbox("Select X-Axis", df.columns.tolist())
        y_axis = st.multiselect("Select Y-Axis Columns", df.select_dtypes(include=['number']).columns)

        if st.button("Generate Plot"):
            fig, ax = plt.subplots(figsize=(10, 6))

            for col in y_axis:
                if plot_type == "Line":
                    sns.lineplot(x=df[x_axis], y=df[col], label=col, ax=ax)
                elif plot_type == "Bar":
                    sns.barplot(x=df[x_axis], y=df[col], ax=ax)
                elif plot_type == "Scatter":
                    sns.scatterplot(x=df[x_axis], y=df[col], label=col, ax=ax)
                elif plot_type == "Box":
                    sns.boxplot(x=df[x_axis], y=df[col], ax=ax)

            ax.set_title(f"{plot_type} Plot: {x_axis} vs {', '.join(y_axis)}")
            ax.set_xlabel(x_axis)
            ax.set_ylabel("Values")
            ax.legend()
            st.pyplot(fig)

            fig.savefig("generated_plot.png")
            with open("generated_plot.png", "rb") as f:
                st.download_button("Download Plot", data=f, file_name="plot.png", mime="image/png")
class AutoML:
    """Handles automated machine learning using PyCaret."""

    def run_automl(self):
        df = st.session_state.cleaned_df
        st.subheader("🤖 AutoML with PyCaret")

        task = st.radio("Select Machine Learning Task", ["Classification", "Regression"])
        target_column = st.selectbox("Select Target Column", df.columns)

        st.session_state.task = task
        st.session_state.target_column = target_column

        if st.button("Run AutoML"):
            st.subheader("🛠 Setting Up AutoML")

            if task == "Classification":
                class_counts = df[target_column].value_counts()
                df = df[df[target_column].map(class_counts) > 1]

                X = df.drop(columns=[target_column])
                y = df[target_column]

                smote = SMOTE()
                X_resampled, y_resampled = smote.fit_resample(X, y)
                df = pd.concat([pd.DataFrame(X_resampled, columns=X.columns),
                                pd.DataFrame(y_resampled, columns=[target_column])],
                               axis=1)

                cls_setup(df, target=target_column, verbose=False)
                best_model = cls_compare()
                st.session_state.final_model = cls_finalize(best_model)

                st.subheader("📊 Model Performance Metrics")
                results = cls_predict(st.session_state.final_model, df)
                st.dataframe(results.describe())

            else:
                reg_setup(df, target=target_column, verbose=False)
                best_model = reg_compare()
                st.session_state.final_model = reg_finalize(best_model)

                st.subheader("📊 Model Performance Metrics")
                results = reg_predict(st.session_state.final_model, df)
                st.dataframe(results.describe())

            st.success(f"✅ Best Model Selected: {str(best_model)}")
            st.session_state.model_ready = True

    def download_model(self):
        if st.session_state.get("model_ready", False):
            st.subheader("📥 Download Trained Model")
            with open("trained_model.pkl", "wb") as f:
                pickle.dump(st.session_state.final_model, f)
            with open("trained_model.pkl", "rb") as f:
                st.download_button("Download Model", data=f,
                                   file_name="trained_model.pkl",
                                   mime="application/octet-stream")

    def make_predictions(self):
        if not st.session_state.get("model_ready", False):
            return

        df = st.session_state.cleaned_df
        task = st.session_state.task
        model = st.session_state.final_model
        target = st.session_state.target_column

        st.subheader("🔮 Make Predictions")
        method = st.radio("Select Prediction Method", ["Manual Input", "Upload Data for Prediction"])

        if method == "Manual Input":
            if "input_data" not in st.session_state:
                st.session_state.input_data = {}

            for col in df.columns:
                if col != target:
                    val = st.text_input(f"Enter value for {col}",
                                        value=st.session_state.input_data.get(col, ""))
                    st.session_state.input_data[col] = val

            if st.button("Predict Outcome"):
                input_df = pd.DataFrame([st.session_state.input_data])
                prediction = (cls_predict(model, input_df) if task == "Classification"
                              else reg_predict(model, input_df))
                st.session_state.predicted_value = prediction.iloc[:, -1].values[0]

        elif method == "Upload Data for Prediction":
            pred_file = st.file_uploader("Upload Dataset for Predictions (CSV)", type=["csv"])
            if pred_file:
                pred_df = pd.read_csv(pred_file)
                prediction = (cls_predict(model, pred_df) if task == "Classification"
                              else reg_predict(model, pred_df))
                pred_df["Prediction"] = prediction.iloc[:, -1]
                st.dataframe(pred_df)

                csv = pred_df.to_csv(index=False)
                st.download_button("Download Predictions", data=csv,
                                   file_name="predictions.csv", mime="text/csv")

        if "predicted_value" in st.session_state:
            st.subheader("🔍 Predicted Outcome")
            st.write(f"**{st.session_state.predicted_value}**")
class DSBot:
    """Handles intelligent querying and visualization using NLP and structured input."""

    def __init__(self):
        self.df = st.session_state.cleaned_df
        self.normalized_columns = {col.lower().strip(): col for col in self.df.columns}
        self.preprocess_data()

    def preprocess_data(self):
        for col in self.df.columns:
            if any(key in col.lower() for key in ["date", "year", "month", "day"]):
                try:
                    self.df[col] = pd.to_datetime(self.df[col], errors='coerce')
                except:
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        st.session_state.cleaned_df = self.df

    def process_question(self):
        st.subheader("💬 Ask DSBot About Your Dataset")
        user_query = st.text_input("Enter your question:")
        if user_query:
            response = self.handle_query(user_query)
            st.write(response)

    def plot_from_app(self, x_col, y_cols, plot_type):
        """Directly generate plot from structured app UI selections."""
        keywords = [plot_type.lower()]
        return self.generate_plot(y_cols, keywords, x_col)

    def handle_query(self, query):
        df = st.session_state.cleaned_df
        query_lower = query.lower().strip()
        doc = nlp(query_lower)

        tokens = [token.text.lower().strip() for token in doc]
        keywords = self.extract_keywords(" ".join(tokens))
        plot_type = next((k for k in ["line", "bar", "scatter", "pie", "histogram"] if k in tokens), None)

        matched_cols = self.extract_columns(doc)
        if matched_cols:
            x_axis = matched_cols[-1]
            y_cols = matched_cols[:-1] if len(matched_cols) > 1 else [matched_cols[0]]
        else:
            y_cols = self.auto_select_columns(keywords)
            x_axis = None

        if plot_type:
            return self.generate_plot(y_cols, keywords, x_axis)
        else:
            return self.analyze_data(y_cols + ([x_axis] if x_axis else []), keywords)

    def extract_columns(self, doc):
        matches = []
        for token in doc:
            text = token.text.lower().strip()
            if text in self.normalized_columns:
                matches.append(self.normalized_columns[text])
            else:
                for norm_col in self.normalized_columns:
                    if text in norm_col:
                        matches.append(self.normalized_columns[norm_col])
                        break
        return list(dict.fromkeys(matches))

    def extract_keywords(self, query):
        keyword_mapping = {
            "mean": ["mean", "average"],
            "max": ["max", "maximum", "highest", "top"],
            "min": ["min", "minimum", "lowest", "bottom"],
            "sum": ["sum", "total"],
            "describe": ["describe", "summary", "overview"],
            "correlation": ["correlation", "relationship", "dependency"],
            "null": ["null", "missing", "empty", "nan"],
            "compare": ["compare", "contrast", "difference"],
            "visualize": ["visualize", "graph", "plot", "chart", "draw"],
            "bar": ["bar", "bar chart"],
            "scatter": ["scatter", "scatterplot"],
            "pie": ["pie", "pie chart"],
            "histogram": ["histogram", "distribution"],
            "line": ["line", "line plot", "trend"]
        }
        return [
            key for key, synonyms in keyword_mapping.items()
            if any(re.search(r"\b" + re.escape(syn) + r"\b", query, re.IGNORECASE) for syn in synonyms)
        ]

    def auto_select_columns(self, keywords):
        df = st.session_state.cleaned_df
        column_mapping = {
            "profit": ["profit"],
            "revenue": ["revenue"],
            "cost": ["cost"],
            "date": ["year", "month", "day", "date"],
            "correlation": list(df.select_dtypes(include=['number']).columns),
            "describe": list(df.columns),
            "visualize": ["profit", "revenue", "cost"],
            "compare": ["profit", "cost"],
            "null": list(df.columns)
        }
        for key in keywords:
            if key in column_mapping:
                return column_mapping[key]
        numeric_cols = list(df.select_dtypes(include=['number']).columns)
        return numeric_cols or list(df.columns)

    def analyze_data(self, columns, keywords):
        df = st.session_state.cleaned_df
        if "mean" in keywords:
            return {col: df[col].mean() for col in columns}
        if "max" in keywords:
            return {col: df[col].max() for col in columns}
        if "min" in keywords:
            return {col: df[col].min() for col in columns}
        if "sum" in keywords:
            return {col: df[col].sum() for col in columns}
        if "describe" in keywords:
            return df[columns].describe()
        if "correlation" in keywords:
            return df[columns].corr()
        if "null" in keywords:
            return df.isnull().sum()
        return "⚠️ Couldn't find a specific task. Ask about stats, correlation, or plots!"

    def generate_plot(self, y_cols, keywords, x_axis=None):
        df = st.session_state.cleaned_df
        fig, ax = plt.subplots(figsize=(8, 5))
        plot_type = next((k for k in ["bar", "scatter", "histogram", "pie", "line"] if k in keywords), "line")

        if not x_axis:
            time_cols = [col for col in df.columns if any(k in col.lower() for k in ["year", "month", "date", "day"])]
            x_axis = time_cols[0] if time_cols else None

        if plot_type == "bar":
            if len(y_cols) >= 2:
                sns.barplot(data=df, x=y_cols[0], y=y_cols[1], ax=ax)
            elif x_axis and y_cols:
                sns.barplot(data=df, x=x_axis, y=y_cols[0], ax=ax)
            else:
                df[y_cols].plot(kind="bar", ax=ax)
            ax.set_title("Bar Chart")

        elif plot_type == "scatter":
            if x_axis and y_cols:
                for col in y_cols:
                    sns.scatterplot(data=df, x=x_axis, y=col, ax=ax, label=col)
            ax.set_title("Scatter Plot")

        elif plot_type == "histogram":
            sns.histplot(df[y_cols[0]], bins=20, kde=True, ax=ax)
            ax.set_title(f"Histogram of {y_cols[0]}")

        elif plot_type == "pie":
            df[y_cols[0]].value_counts().plot.pie(autopct='%1.1f%%', ax=ax)
            ax.set_ylabel("")
            ax.set_title(f"Pie Chart of {y_cols[0]}")

        else:  # line plot
            for col in y_cols:
                sns.lineplot(data=df, x=x_axis or df.index, y=col, ax=ax, label=col)
            ax.set_title(f"Line Plot of {', '.join(y_cols)}")

        plt.xlabel(x_axis or "Index")
        plt.ylabel("Value")
        plt.legend()
        st.pyplot(fig)
def main():
    """Fully Interactive EDA, Cleaning, Visualization & Modeling App"""

    st.title("📊 Interactive EDA, Cleaning, Visualization & Modeling App")

    # Upload dataset once
    data = st.file_uploader("Upload Your Dataset (CSV, XLSX)", type=["csv", "xlsx"])

    if data and "cleaned_df" not in st.session_state:
        try:
            if data.name.endswith(".csv"):
                df = pd.read_csv(data)
            else:
                df = pd.read_excel(data)
            st.session_state.cleaned_df = df.copy()
            st.success("✅ Dataset uploaded successfully!")
        except Exception as e:
            st.error(f"❌ Failed to load dataset: {e}")
            return

    if "cleaned_df" not in st.session_state:
        st.info("📂 Please upload a dataset to begin.")
        return

    # Sidebar navigation
    activities = ["EDA", "Cleaning", "Data Visualization", "AutoML", "DSBot"]
    choice = st.sidebar.selectbox("Select Activity", activities)

    if choice == "EDA":
        st.subheader("🧠 Exploratory Data Analysis")
        eda = ExploratoryDataAnalysis()
        eda.display_head()
        eda.show_shape()
        eda.show_columns()
        eda.summary_statistics()
        eda.missing_values()
        eda.correlation_analysis()
        eda.feature_distribution()
        eda.outlier_detection()

    elif choice == "Cleaning":
        st.subheader("🛠 Data Cleaning Tools")
        clean = Cleaning()
        clean.remove_duplicates()
        clean.drop_columns()
        clean.handle_missing()
        clean.scale_columns()
        clean.encode_categoricals()
        clean.download_data()
        clean.reset()

    elif choice == "Data Visualization":
        st.subheader("📊 Custom Plot Generation")
        viz = DataVisualization()
        viz.generate_plot()

    elif choice == "AutoML":
        st.subheader("🤖 Automated Machine Learning")
        automl = AutoML()
        automl.run_automl()
        automl.download_model()
        automl.make_predictions()

    elif choice == "DSBot":
        st.subheader("💬 Ask DSBot Anything")
        dsbot = DSBot()
        dsbot.process_question()
if __name__ == "__main__":
    main()
