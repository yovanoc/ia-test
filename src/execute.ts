import * as tf from "@tensorflow/tfjs";
import { saveChart } from "./saveChart.js";
import {
  addMsToDate,
  timeFrameToMS,
  generateNextPrediction,
  minMaxInverseScaler,
  minMaxScaler,
  processData,
} from "./helpers.js";
import { getData, Symbol, TimeFrame } from "./data.js";
import { Model } from "./models/model.js";

export const execute = async ({
  modelTmp,
  symbol,
  timeFrame,
  windowSize,
  trainEpochs,
  shouldTrain = false
}: {
  modelTmp: Model;
  windowSize: number;
  trainEpochs: number;
  symbol: Symbol;
  timeFrame: TimeFrame;
  shouldTrain: boolean;
}) => {
  const data = await getData(symbol, timeFrame);
  const splitDate = new Date("2021-01-01T22:00:00.000Z");
  // Process the data and create the train sets
  const results = processData(data, windowSize, splitDate);
  // Get the datetime labels use in graph
  const labels = data.splice(-results.visY.length + 1).map((val) => val.closeTime);
  // Crate the set for stock price prediction for the next entry
  const nextPrediction = generateNextPrediction(
    results.originalData,
    results.timePortion
  );
  // Get the last date from the data set
  const predictDate = addMsToDate(labels[labels.length - 1], timeFrameToMS(timeFrame));
  // Remember the min and max in order to revert (min-max scaler) the scaled data later
  const max = results.max;
  const min = results.min;
  // Train the model using the tensor data
  // Repeat multiple epochs so the error rate is smaller (better fit for the data)
  if (shouldTrain) {
    // Transform the data to tensor data
    // Reshape the data in neural network input format [number_of_samples, timePortion, 1];
    const trainTensorData = {
      tensorTrainX: tf
        .tensor1d(results.trainX)
        .reshape([results.trainSize, results.timePortion, 1]),
      tensorTrainY: tf.tensor1d(results.trainY),
    };
    await modelTmp.train(trainTensorData, trainEpochs);
    await modelTmp.save();
  } else {
    await modelTmp.load();
  }
  const model = modelTmp.model;
  // Predict for the same train data
  // We gonna show the both (original, predicted) sets on the graph
  // so we can see how well our model fits the data
  const visTensorData = {
    tensorVisX: tf
      .tensor1d(results.visX)
      .reshape([results.visSize, results.timePortion, 1]),
    tensorVisY: tf.tensor1d(results.visY),
  };
  const predictedX = model.predict(
    visTensorData.tensorVisX
  ) as tf.Tensor<tf.Rank>;
  // Scale the next day features
  const nextDayPredictionScaled = minMaxScaler(nextPrediction, min, max);
  // Transform to tensor data
  const tensorNextDayPrediction = tf
    .tensor1d(nextDayPredictionScaled.data)
    .reshape([1, results.timePortion, 1]);
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
  const trainYInverse = minMaxInverseScaler(results.visY, min, max);
  // Plot the original (trainY) and predicted values for the same features set (trainX)
  await saveChart(
    symbol,
    trainYInverse.data as number[],
    predictedXInverse.data,
    labels.map((i) => i.toLocaleString())
  );
};
