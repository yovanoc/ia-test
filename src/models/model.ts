import * as tf from "@tensorflow/tfjs";

export type BuildFn = () => tf.Sequential;

export abstract class Model {
  #link: string;

  model: tf.Sequential | tf.LayersModel;


  constructor(link: string) {
    this.#link = link;
    this.model = this.build();
  }

  abstract build(): tf.Sequential;

  public async train(data: {
    tensorTrainX: tf.Tensor<tf.Rank>;
    tensorTrainY: tf.Tensor1D;
  }, epochs: number) {
    console.log("MODEL SUMMARY: ")
    this.model.summary();

    // Optimize using adam (adaptive moment estimation) algorithm
    this.model.compile({ optimizer: tf.train.adam(), loss: "meanSquaredError" });

    // Train the model
    const result = await this.model.fit(data.tensorTrainX, data.tensorTrainY, {
      epochs,
      verbose: 1,
      shuffle: true
    })

    /*for (let i = result.epoch.length-1; i < result.epoch.length; ++i) {
        print("Loss after Epoch " + i + " : " + result.history.loss[i]);
    }*/
    console.log("Loss after last Epoch (" + result.epoch.length + ") is: " + result.history.loss[result.epoch.length - 1]);
  }

  public save() {
    return this.model.save(this.#link);
  }

  public async load(): Promise<void> {
    this.model = await tf.loadLayersModel(`${this.#link}/model.json`);
  }
}