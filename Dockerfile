# Use the official Python image from the Docker Hub  
FROM python:3.11
  
# Set the working directory in the container  
WORKDIR /app  
  
# Copy the current directory contents into the container at /app  
COPY . /app  
  
# Install any needed packages specified in requirements.txt  
RUN pip install --no-cache-dir -r requirements.txt  
  
# Expose port 8000 to the world outside this container  
EXPOSE 8000  
  
# Run the FastAPI application using uvicorn  
CMD ["uvicorn", "appmongo:app", "--host", "0.0.0.0", "--port", "8000"]

#uvicorn app:app --reload