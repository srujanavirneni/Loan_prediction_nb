FROM python:3.12

# Set the working directory
WORKDIR /code

# Copy requirements and install
COPY ./requirements.txt /code/requirements.txt
RUN pip install -r /code/requirements.txt

# Copy dataset, script, and app directory
COPY ./loan_prediction.csv /code/
COPY ./model_prediction.py /code/model_prediction.py
COPY ./app /code/app

# Run the model generation script
RUN python model_prediction.py

# Move the generated model file into the app directory
RUN mv rfr_v1.pkl app/rfr_v1.pkl

# Remove the dataset
RUN rm /code/loan_prediction.csv

# Set the working directory to the app folder
WORKDIR /code/app

# Run the FastAPI application
CMD ["fastapi", "run", "main.py", "--port", "80"]
