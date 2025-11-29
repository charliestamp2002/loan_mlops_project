
FROM python:3.10

# working directory inside the container
WORKDIR /app

# Copy requirements first
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of project into the container
COPY . .

# Expose port 
EXPOSE 8000

# Command to run the FastAPI app with Uvicorn
CMD ["uvicorn", "loan_mlops.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]