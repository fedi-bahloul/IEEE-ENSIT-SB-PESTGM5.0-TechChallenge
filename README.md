# Intelligent Energy Management System for Smart House

This project is an Intelligent Energy Management System (EMS) designed for smart houses. It analyzes solar energy generation and consumption across the house and its components by retrieving data from sensors and IoT devices. The results and predictions are displayed in a user-friendly interface.

## Project Structure

The project is divided into two main parts: the AI part and the UI part.

### AI Part

#### Datasets

1. **Homec14.csv**: This dataset is used for training the model. It contains the following columns:
    - `time`
    - `use [kW]`
    - `gen [kW]`
    - `House overall [kW]`
    - `Dishwasher [kW]`
    - `Furnace 1 [kW]`
    - `Furnace 2 [kW]`
    - `Home office [kW]`
    - `Fridge [kW]`
    - `Garage door [kW]`
    - `Kitchen [kW]`
    - `Kitchenette [kW]`
    - `Well [kW]`
    - `Microwave [kW]`
    - `Living room [kW]`
    - `Solar [kW]`
    - `temperature`
    - `icon`
    - `humidity`
    - `visibility`
    - `summary`
    - `apparentTemperature`
    - `pressure`
    - `windSpeed`
    - `cloudCover`
    - `windBearing`
    - `precipIntensity`
    - `dewPoint`
    - `precipProbability`

2. **SmarthHouse.csv**: This dataset is derived from Homec14.csv and used for specific analyses.
3. **EXTRACT.csv**: This dataset is used for plotting diagrams in the user interface.

#### Scripts

- **EMS.ipynb**: Jupyter Notebook containing intelligent models using Random Forest and ARIMA for prediction and data analysis.
- **DataExtraction.ipynb**: Jupyter Notebook used for creating the EXTRACT.csv file.
- **EMS.py**: Python script to integrate the model into the Dash application (app.py).

### UI Part

- **app.py**: Contains the Dash framework application, which provides the user interface for visualizing data, results, and predictions.

### Arduino Part

This part includes three files for handling various sensors in the house:

1. **jardin.ino**: Manages garden sensors and sends data to the backend server.
2. **interphoneinteligne.ino**: Handles the smart interphone.
3. **codefinal.ino**: Manages the main house sensors and sends data to the backend server.

## Installation and Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/fedi-bahloul/IEEE-ENSIT-SB-PESTGM5.0-TechChallenge
