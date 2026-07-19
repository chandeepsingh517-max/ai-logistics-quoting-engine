# 1. Use the official lightweight Python image
FROM python:3.10-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy your actual code and database into the container
COPY . .

# 5. Expose the port FastAPI uses
EXPOSE 8000

# 6. The command to start the server when the container boots
CMD ["uvicorn", "capstone_api:app", "--host", "0.0.0.0", "--port", "8000"]