import * as tf from "@tensorflow/tfjs";
import { TIME_PORTION } from "../index.js";
import { Model } from "./model.js";

export class CnnModel extends Model {
  build() {
    // Linear (sequential) stack of layers
    const model = tf.sequential();

    // Define input layer
    model.add(tf.layers.inputLayer({
      inputShape: [TIME_PORTION, 1],
    }));

    // Add the first convolutional layer
    model.add(tf.layers.conv1d({
      kernelSize: 2,
      filters: 128,
      strides: 1,
      useBias: true,
      activation: 'relu',
      kernelInitializer: 'VarianceScaling'
    }));

    // Add the Average Pooling layer
    model.add(tf.layers.averagePooling1d({
      poolSize: [2],
      strides: [1]
    }));

    // Add the second convolutional layer
    model.add(tf.layers.conv1d({
      kernelSize: 2,
      filters: 64,
      strides: 1,
      useBias: true,
      activation: 'relu',
      kernelInitializer: 'VarianceScaling'
    }));

    // Add the Average Pooling layer
    model.add(tf.layers.averagePooling1d({
      poolSize: [2],
      strides: [1]
    }));

    // Add Flatten layer, reshape input to (number of samples, number of features)
    model.add(tf.layers.flatten({

    }));

    // Add Dense layer, 
    model.add(tf.layers.dense({
      units: 1,
      kernelInitializer: 'VarianceScaling',
      activation: 'linear'
    }));

    return model;
  }
}
