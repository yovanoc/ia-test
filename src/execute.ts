import * as tf from "@tensorflow/tfjs";
import { saveChart } from "./chart.js";
import {
  addDays,
  generateNextDayPrediction,
  minMaxInverseScaler,
  minMaxScaler,
  processData,
} from "./helpers.js";
import { getData } from "./data.js";
import { Model } from "./models/model.js";
import { TIME_PORTION } from "./index.js";

const epochs = 100;

export const execute = async (modelTmp: Model, shouldTrain = false) => {
  const symbol = "BTCUSDT";
  const allData = await getData(symbol);
  const splitDate = new Date("2021-01-01T22:00:00.000Z");
  const trainData = allData.filter((d) => new Date(d.date) < splitDate);
  const visData = allData.filter((d) => new Date(d.date) >= splitDate);
  // Get the datetime labels use in graph
  const labels = visData.map((val) => new Date(val.date));
  // Process the data and create the train sets
  const trainResult = processData(trainData, TIME_PORTION);
  const visResult = processData(visData, TIME_PORTION);
  // Crate the set for stock price prediction for the next day
  const nextDayPrediction = generateNextDayPrediction(
    visResult.originalData,
    visResult.timePortion
  );
  // Get the last date from the data set
  const predictDate = addDays(labels[labels.length - 1], 1);
  // Transform the data to tensor data
  // Reshape the data in neural network input format [number_of_samples, timePortion, 1];
  const trainTensorData = {
    tensorTrainX: tf
      .tensor1d(trainResult.trainX)
      .reshape([trainResult.size, trainResult.timePortion, 1]),
    tensorTrainY: tf.tensor1d(trainResult.trainY),
  };
  // Remember the min and max in order to revert (min-max scaler) the scaled data later
  const max = visResult.max;
  const min = visResult.min;
  // Train the model using the tensor data
  // Repeat multiple epochs so the error rate is smaller (better fit for the data)
  if (shouldTrain) {
    await modelTmp.train(trainTensorData, epochs);
    await modelTmp.save();
  } else {
    await modelTmp.load();
  }
  const model = modelTmp.model;
  // Predict for the same train data
  // We gonna show the both (original, predicted) sets on the graph
  // so we can see how well our model fits the data
  const visTensorData = {
    tensorTrainX: tf
      .tensor1d(visResult.trainX)
      .reshape([visResult.size, visResult.timePortion, 1]),
    tensorTrainY: tf.tensor1d(visResult.trainY),
  };
  const predictedX = model.predict(
    visTensorData.tensorTrainX
  ) as tf.Tensor<tf.Rank>;
  // Scale the next day features
  const nextDayPredictionScaled = minMaxScaler(nextDayPrediction, min, max);
  // Transform to tensor data
  const tensorNextDayPrediction = tf
    .tensor1d(nextDayPredictionScaled.data)
    .reshape([1, visResult.timePortion, 1]);
  // Predict the next day stock price
  const predictedValue = model.predict(
    tensorNextDayPrediction
  ) as tf.Tensor<tf.Rank>;
  // Get the predicted data for the train set
  const predValue = await predictedValue.data();
  // Revert the scaled features, so we get the real values
  const inversePredictedValue = minMaxInverseScaler(predValue, min, max);
  // Print the predicted stock price value for the next day
  console.log(
    "Predicted Stock Price of " +
      symbol +
      " for date " +
      predictDate.toLocaleString() +
      " is: " +
      inversePredictedValue.data[0].toFixed(3) +
      "$"
  );
  // Get the next day predicted value
  const pred = await predictedX.data();
  // Revert the scaled feature
  const predictedXInverse = minMaxInverseScaler(pred, min, max);
  // Convert Float32Array to regular Array, so we can add additional value
  predictedXInverse.data = Array.from(predictedXInverse.data);
  // Add the next day predicted stock price so it's showed on the graph
  predictedXInverse.data[predictedXInverse.data.length] =
    inversePredictedValue.data[0];
  // Revert the scaled labels from the trainY (original),
  // so we can compare them with the predicted one
  const trainYInverse = minMaxInverseScaler(visResult.trainY, min, max);
  // Plot the original (trainY) and predicted values for the same features set (trainX)
  await saveChart(
    symbol,
    trainYInverse.data as number[],
    predictedXInverse.data,
    labels.map((i) => i.toLocaleString())
  );
};
