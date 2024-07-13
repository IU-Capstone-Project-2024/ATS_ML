# Automatic Trade System - ML Model

Welcome to the Automatic Trade System (ATS) project! This repository is dedicated to developing an ML component of an automated trading system.

## Table of Contents

1. [Overview](#overview)
2. [Stack](#stack)
3. [Usage](#usage)
4. [Progress Report](#progress-report)
5. [Issues and Support](#issues-and-support)
6. [License](#license)
7. [Authors](#authors)

## Overview

The ATS project aims to develop an automated trading system that leverages machine learning algorithms to predict market trends and make informed trading decisions. The project consists of a frontend web interface and a backend machine learning component.

This repository houses an autonomous MLops system centered on the Long Short-Term Memory network (LSTM) model, which predicts the future price of the selected cryptocurrency for the next 15 seconds time interval based on incoming real-time information. Afterward, the predicted information is transferred to the automated trading system through the API interface, where the trading decision-making process is executed based on the received data.

## Stack

### Python libraries

![Pandas](https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white)
![Numpy](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![OpenAI Gym](https://img.shields.io/badge/openai%20gym-blue?style=for-the-badge&logo=openai%20gym)


* **Pandas**, **NumPy**: fundamental for data manipulation and numerical operations.

* **Matplotlib**: essential for creating visualizations.

* **TensorFlow**, **PyTorch**, **Gymnasium**: powerful tools for building and training machine learning and deep learning models.

## Usage
1. Clone the repository using the following command:

 ```bash
git clone https://github.com/IU-Capstone-Project-2024/ATS_ML.git
```

2. Install dependencies:

 ```bash
pip install -r requirements.txt
```

3. Set up the Environment:

  Create a virtual environment for the project and activate it. This will help isolate the project dependencies and ensure a clean installation.

## Performance

**What we have tested and what results we have achieved:**

1. Test runs of trading algorithms were executed successfully on the exchange.
2. We established and tested an environment for our AI agent to continue training it using reinforcement techniques.

**ML achivments during testing process:**

The behaviour of agent before training using reinforcement techniques:

![statistics](https://capstone.innopolis.university/2024/ATS/ml1.jpeg)

After training:

![statistics](https://capstone.innopolis.university/2024/ATS/ml2.jpeg)

## Project timeline and progress reports:

[Innopolis University Summer course 2024 Capstone project](https://capstone.innopolis.university/docs/2024/ats)



## Issues and Support

If you encounter any issues or need support, please open an issue in the GitHub repository.

## License

The ATS project is licensed under the MIT License. For more details, see the `LICENSE` file.

## Authors

- [Ivan Golov](https://github.com/IVproger): Team Lead
- [Daniil Abrosimov](https://github.com/abrosimov-software): ML Engineer
- [Dmitriy Nekrasov](https://github.com/YouOnlyLive1ce): ML Engineer
