# Use the official Python image from the Docker Hub  
FROM python:3.12
  
# Set the working directory in the container  
WORKDIR /app  
  
# Copy the current directory contents into the container at /app  
COPY . /app  
  
# Install any needed packages specified in requirements.txt  
RUN pip install --no-cache-dir -r requirements.txt  
  
# Expose port 80 to the world outside this container  
EXPOSE 80
  
# Run the FastAPI application using uvicorn  
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]

#uvicorn app:app --reload