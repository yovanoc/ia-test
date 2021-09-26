import * as tf from "@tensorflow/tfjs";
import { Model } from "./model.js";

export class RnnModel extends Model {
  override build(windowSize: number) {
    const model = tf.sequential();

    // input dense layer
    const input_layer_shape = windowSize;
    const input_layer_neurons = 64;

    // LSTM
    const rnn_input_layer_features = 16;
    const rnn_input_layer_timesteps = input_layer_neurons / rnn_input_layer_features;
    const rnn_input_shape = [rnn_input_layer_features, rnn_input_layer_timesteps]; // the shape have to match input layer's shape
    const rnn_output_neurons = 16; // number of neurons per LSTM's cell

    // output dense layer
    const output_layer_shape = rnn_output_neurons; // dense layer input size is same as LSTM cell
    const output_layer_neurons = 1; // return 1 value

    model.add(tf.layers.dense({ units: input_layer_neurons, inputShape: [input_layer_shape, 1] }));
    model.add(tf.layers.reshape({ targetShape: rnn_input_shape }));

    const lstmCells = new Array<ReturnType<typeof tf.layers.lstmCell>>();
    for (let index = 0; index < 4; index++) {
      lstmCells.push(tf.layers.lstmCell({ units: rnn_output_neurons }));
    }

    model.add(tf.layers.rnn({
      cell: lstmCells,
      inputShape: rnn_input_shape,
      returnSequences: false
    }));

    model.add(tf.layers.dense({ units: output_layer_neurons, inputShape: [output_layer_shape] }));
    return model;
  }
}
