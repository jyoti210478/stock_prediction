# LSTM Model for Stock Price Prediction

This project implements an LSTM (Long Short-Term Memory) neural network model for stock price prediction. The model is trained using historical data and is capable of making predictions on unseen data.

## Table of Contents
- [Description](#description)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Description

The LSTM model uses historical stock price data to predict future stock prices. It preprocesses the data, trains the model, and saves the trained model for future use.

## Installation

1. Clone the repository to your local machine:
```
 git clone https://github.com/Chetan7595/Stock-Price-Prediction.git
```


2. Install the required dependencies:
pip install Pandas, Numpy, Scikit-learn, Tensorflow

## Usage

1. Ensure you have the following files in the project directory:
- `sample_input.csv`: CSV file containing the historical stock price data. It should include a column named 'Close' representing the closing prices.

2. Open the `LSTM_MODEL.ipynb` notebook in Jupyter Notebook or any compatible environment.

3. Run the notebook cells sequentially to execute the code.

4. The code preprocesses the data, splits it into training and validation sets, and trains the LSTM model using the training data.

5. Adjust the model architecture, hyperparameters, or data preprocessing steps as needed.

6. After training, the model will be saved as `lstm_Model.h5` in the project directory.

7. Use the trained model to make predictions on unseen data or further evaluate its performance.

## Contributing

Contributions to this project are welcome. To contribute, please follow these steps:

1. Fork the repository.

2. Create a new branch.

3. Make your changes and commit them.

4. Push the changes to your forked repository.

5. Submit a pull request.
