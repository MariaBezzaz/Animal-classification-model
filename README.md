# Animal Prediction Model

This project is a web application that predicts the type of animal in an uploaded image 

## Project Overview

The application allows users to upload an image of an animal and get a prediction of the animal type. The model can classify 90 different animal species.

## Setup

### Prerequisites

- Python 3.7 or higher
- Flask
- TensorFlow
- Pillow
- NumPy

### Installation

1. Clone the repository:

    ```sh
    git clone https://github.com/yourusername/animal-prediction-model.git
    cd animal-prediction-model
    ```

2. Create a virtual environment and activate it:

    ```sh
    python -m venv env
    source env/bin/activate  # On Windows use `env\Scripts\activate`
    ```


3. Ensure the `inceptionv3_model.h5` file is in the project directory.

## Usage

1. Run the Flask application:

    ```sh
    python app.py
    ```

2. Open a web browser and go to `http://127.0.0.1:5000`.

3. Upload an image using the form and click "Upload".

4. The prediction results will be displayed on the page.

## File Structure

project/
│
├── static/
│ ├── css/
│ │ └── styles.css
│
├── templates/
│ ├── index.html
│ └── result.html
│
├── app.py
├── inceptionv3_model.h5
├── model.ipynb
