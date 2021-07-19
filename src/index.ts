import "@tensorflow/tfjs-node";
import { execute } from "./execute.js";
import { CnnModel } from "./models/cnn.js";

export const TIME_PORTION = 30;

const main = async () => {
  const model = new CnnModel("file://./models/cnn");
  await execute(model, true);
};

main();