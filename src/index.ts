import "@tensorflow/tfjs-node";
import { execute } from "./execute.js";
import { CnnModel } from "./models/cnn.js";

const windowSize = 168;
const trainEpochs = 20;

const main = async () => {
  const model = new CnnModel({
    link: "file://./models/cnn",
    windowSize,
  });
  await execute({
    modelTmp: model,
    windowSize,
    trainEpochs,
    shouldTrain: true,
    symbol: "BTCUSDT",
    timeFrame: "1h"
  });
};

main();