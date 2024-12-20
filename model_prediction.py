import csv
import numpy as np
import pandas as pd
import sqlite3

from sklearn.compose import ColumnTransformer
from sklearn.discriminant_analysis import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FunctionTransformer, Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

def create_tables(conn):
    # SQL commands to create the necessary tables
    create_table_queries = [
        "DROP TABLE IF EXISTS Personal_Details",
        "DROP TABLE IF EXISTS Loan",
        "DROP TABLE IF EXISTS Prop_Details",
        "DROP TABLE IF EXISTS Edu_Details",
        """
        CREATE TABLE IF NOT EXISTS [Prop_Details] (
            [Prop_details_ID] INTEGER PRIMARY KEY NOT NULL,
            [Property_Area] TEXT NOT NULL
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS [Edu_Details] (
            [Edu_details_ID] INTEGER PRIMARY KEY NOT NULL,
            [Education] TEXT NOT NULL
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS [Loan] (
            [ID] INTEGER PRIMARY KEY NOT NULL,
            [Loan_ID] TEXT NOT NULL,
            [ApplicantIncome] INTEGER NOT NULL,
            [CoapplicantIncome] INTEGER NOT NULL,
            [LoanAmount] INTEGER,
            [Loan_Amount_Term] INTEGER,
            [Credit_History] INTEGER,
            [Loan_Status] TEXT NOT NULL
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS [Personal_Details] (
            [Personal_details_ID] INTEGER PRIMARY KEY NOT NULL,
            [Loan_ID] TEXT NOT NULL,
            [Gender] TEXT,
            [Married] TEXT,
            [Dependents] TEXT,
            [Edu_details_ID] TEXT NOT NULL,
            [Self_Employed] TEXT,
            [Prop_details_ID] TEXT NOT NULL,
            FOREIGN KEY(Loan_ID) REFERENCES Loan(ID),
            FOREIGN KEY(Edu_details_ID) REFERENCES Edu_Details(Edu_details_ID),
            FOREIGN KEY(Prop_details_ID) REFERENCES Prop_Details(Prop_details_ID)
        );
        """
    ]

    for query in create_table_queries:
        conn.execute(query)
    conn.commit()


def insert_data(conn, table_name, columns, values):
    """
    Insert data into a table, handling duplicate entries gracefully.
    """
    insert_query = f"INSERT OR IGNORE INTO {table_name} ({', '.join(columns)}) VALUES ({', '.join(['?' for _ in columns])})"
    conn.executemany(insert_query, values)
    conn.commit()


def load_data_from_csv(file_path):
    """
    Load and preprocess the CSV data.
    """
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)[1:]

        data = [dict(zip(header, row[1:])) for row in reader]
    
    return data


def get_unique_values(df, column_name):
    """
    Extract unique values for given column from the DataFrame.
    """
    return set(row[column_name] for row in df)

def main():
    conn = sqlite3.connect("loan_requests.db")
    conn.execute("PRAGMA foreign_keys = 1;")

    create_tables(conn)

    df = load_data_from_csv('loan_prediction.csv')

    prop_values = [(area,) for area in get_unique_values(df, "Property_Area")]
    edu_values = [(edu,) for edu in get_unique_values(df, "Education")]

    loan_details_col = ['ID', 'Loan_ID', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Loan_Status']
    loan_details = [
        (
            idx + 1,
            row["Loan_ID"],
            int(row["ApplicantIncome"]),
            float(row["CoapplicantIncome"]),
            int(row["LoanAmount"]) if row["LoanAmount"] else None,
            int(row["Loan_Amount_Term"]) if row["Loan_Amount_Term"] else None,
            int(row["Credit_History"]) if row["Credit_History"] else None,
            row["Loan_Status"]
        )
        for idx, row in enumerate(df)
    ]

    insert_data(conn, "Prop_Details", ["Property_Area"], prop_values)
    insert_data(conn, "Edu_Details", ["Education"], edu_values)
    insert_data(conn, "Loan", loan_details_col, loan_details)


    prop_dict = {row[1]: row[0] for row in conn.execute("SELECT * FROM Prop_Details")}
    edu_dict = {row[1]: row[0] for row in conn.execute("SELECT * FROM Edu_Details")}
    loan_dict = {row[1]: row[0] for row in conn.execute("SELECT ID, Loan_ID FROM Loan")}

    personal_details = [
        (
            idx + 1,
            loan_dict[row["Loan_ID"]],
            row["Gender"] if row["Gender"] else None,
            row["Married"] if row["Married"] else None,
            row["Dependents"] if row["Dependents"] else None,
            edu_dict[row["Education"]],
            row["Self_Employed"] if row["Self_Employed"] else None,
            prop_dict[row["Property_Area"]]
        )
        for idx, row in enumerate(df)
    ]

    insert_data(conn, "Personal_Details", ['Personal_details_ID', 'Loan_ID', 'Gender', 'Married', 'Dependents', 'Edu_details_ID', 'Self_Employed', 'Prop_details_ID'], personal_details)

    select_query = """
            select l.Loan_ID, p.Gender, p.Married, p.Dependents, e.Education, p.Self_Employed, l.ApplicantIncome, l.CoapplicantIncome, l.LoanAmount, l.Loan_Amount_Term, l.Credit_History, prop.Property_Area, l.Loan_Status
            from Personal_Details as p
            inner join Loan as l on l.ID = p.Loan_ID
            inner join Edu_Details as e on e.Edu_details_ID = p.Edu_details_ID
            inner join Prop_Details as prop on prop.Prop_details_ID = p.Prop_details_ID;
        """

    result = pd.read_sql_query(select_query, conn)

    conn.close()

    return result

NUMERIC_FEATURES = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History"]
CATEGORICAL_FEATURES = ["Gender", "Married", "Education", "Self_Employed", "Property_Area"]


loan_app = main()
loan_app = loan_app.drop('Loan_ID', axis=1)
loan_app['Total_Income'] = loan_app['ApplicantIncome'] + loan_app['CoapplicantIncome']
loan_app['Loan_Status'] = loan_app['Loan_Status'].map({'Y': 1, 'N': 0})
loan_app[NUMERIC_FEATURES] = loan_app[NUMERIC_FEATURES].astype(float)

loan_app[NUMERIC_FEATURES] = loan_app[NUMERIC_FEATURES].astype(float)
loan_app["Credit_History"] = loan_app["Credit_History"].fillna(2.0)
loan_app["Credit_History_Cat"] = pd.cut(
    loan_app["Credit_History"], 
    bins=[-np.inf, 0, 1, 2], 
    labels=[1, 2, 3]
)

strat_train_set, strat_test_set = train_test_split(
    loan_app, test_size=0.20, stratify=loan_app["Credit_History_Cat"], random_state=42
)

for set_ in (strat_train_set, strat_test_set):
    set_.drop("Credit_History_Cat", axis=1, inplace=True)

X_train = strat_train_set.drop("Loan_Status", axis=1)
y_train = strat_train_set["Loan_Status"].copy()
X_test = strat_test_set.drop("Loan_Status", axis=1)
y_test = strat_test_set["Loan_Status"].copy()

NUMERIC_FEATURES_FE = NUMERIC_FEATURES + ["Total_Income"]


numeric_pipeline = make_pipeline(
    SimpleImputer(strategy="median"),
    FunctionTransformer(np.log1p),
    StandardScaler(),
    MinMaxScaler()
)

categorical_pipeline = make_pipeline(
    SimpleImputer(strategy="most_frequent"),
    OneHotEncoder(handle_unknown="ignore")
)

preprocessing = Pipeline([
    ('col_transform', ColumnTransformer(
        [
            ("num", numeric_pipeline, NUMERIC_FEATURES_FE),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES)
        ],
        remainder='drop'  # ensure no leftover columns with NaNs
    ))
])

Ridge_pipeline = Pipeline([
    ('preprocessing', preprocessing),
    ('clf', RidgeClassifier(random_state=42))
])

from sklearn.model_selection import GridSearchCV

param_grid = {
    'clf__alpha': [0.1, 1.0, 10.0]
}

grid_search = GridSearchCV(Ridge_pipeline, param_grid, cv=3, scoring='f1')
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

housing_predictions = best_model.predict(X_train)
orig = root_mean_squared_error(y_train, housing_predictions)
print("Original Model:", orig)

import dill
with open('rfr_v1.pkl', 'wb') as f:
    dill.dump(best_model, f)

with open('rfr_v1.pkl', 'rb') as f:
    reloaded_model = dill.load(f)

reloaded = root_mean_squared_error(y_train, reloaded_model.predict(X_train))
print("Reloaded Model:", reloaded)
