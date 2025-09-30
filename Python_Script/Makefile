.PHONY: all clean install train-pipeline data-pipeline streaming-inference run-all help

# Let Make keep using cmd.exe (default)
# No ACTIVATE needed
PYTHON := .\.venv\Scripts\python.exe
PIP := .\.venv\Scripts\pip.exe
MLFLOW_PORT := 5001


all: help

help:
	@echo "Available targets:"
	@echo "  make install             - Install project dependencies and set up environment"
	@echo "  make data-pipeline       - Run the data pipeline"
	@echo "  make train-pipeline      - Run the training pipeline"
	@echo "  make streaming-inference - Run the streaming inference pipeline with the sample JSON"
	@echo "  make run-all             - Run all pipelines in sequence"
	@echo "  make clean               - Clean up artifacts"

install:
	@echo "Creating virtual environment..."
	python -m venv .venv
	@echo "Upgrading pip and installing requirements..."
	$(PYTHON) -m pip install --upgrade pip
	$(PIP) install -r requirements.txt
	@echo "Installation completed!"
	@echo "To activate manually in PowerShell: . .\\.venv\\Scripts\\Activate.ps1"

clean:
	@echo "Cleaning up artifacts..."
	if exist "artifacts\\models"       del /q /s "artifacts\\models\\*"
	if exist "artifacts\\evaluation"   del /q /s "artifacts\\evaluation\\*"
	if exist "artifacts\\predictions"  del /q /s "artifacts\\predictions\\*"
	if exist "data\\processed"         del /q /s "data\\processed\\*"
	@echo "Cleanup completed!"

data-pipeline:
	@echo "Running data pipeline..."
	$(PYTHON) pipelines\\data_pipeline.py

train-pipeline:
	@echo "Running training pipeline..."
	$(PYTHON) pipelines\\training_pipeline.py

streaming-inference:
	@echo "Running streaming inference pipeline with sample JSON..."
	$(PYTHON) pipelines\\streaming_inference_pipeline.py

run-all:
	@echo "Running all pipelines in sequence..."
	$(PYTHON) pipelines\\data_pipeline.py
	$(PYTHON) pipelines\\training_pipeline.py
	$(PYTHON) pipelines\\streaming_inference_pipeline.py
	@echo "All pipelines completed successfully!"


mlflow-ui:
	@echo "Launching MLflow UI..."
	@echo "MLflow UI will be available at: http://localhost:$(MLFLOW_PORT)"
	@powershell -NoProfile -Command Start-Process "http://localhost:$(MLFLOW_PORT)"
	$(PYTHON) -m mlflow ui --host 127.0.0.1 --port $(MLFLOW_PORT) 2>&1

stop-mlflow:
	@echo "Stopping all MLflow servers..."
	-@for /f "tokens=5" %%P in ('netstat -ano ^| findstr LISTENING ^| findstr :$(MLFLOW_PORT)') do @echo Killing PID %%P & taskkill /PID %%P /F >nul 2>nul
	@echo "✅ All MLflow servers have been stopped"
