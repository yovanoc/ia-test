export type Data = {
  date: Date;
  price: number;
}

export type LoadData = {
  date: string;
  price: string;
}

export const convertData = (data: LoadData): Data => ({
  date: new Date(data.date),
  price: parseFloat(data.price)
});

/*
    Process the price data
    Create the train features and labels for cnn
    Each prediction is base on previous timePortion days
    ex. timePortion=7, prediction for the next day is based to values of the previous 7 days
*/
export const processData = (data: Data[], timePortion: number) => {
  const trainX = new Array<number>();
  const trainY = new Array<number>();
  const size = data.length;

  const features = new Array<number>();
  for (let i = 0; i < size; i++) {
    features.push(data[i].price);
  }

  // Scale the values
  const { min, max } = minMax2DArray(features);
  const scaledData = minMaxScaler(features, min, max);
  const scaledFeatures = scaledData.data;

  // Create the train sets
  for (let i = timePortion; i < size; i++) {

    for (let j = (i - timePortion); j < i; j++) {
      trainX.push(scaledFeatures[j]);
    }

    trainY.push(scaledFeatures[i]);
  }

  return {
    size: (size - timePortion),
    timePortion: timePortion,
    trainX: trainX,
    trainY: trainY,
    min: scaledData.min,
    max: scaledData.max,
    originalData: features,
  }
};


/*
  This will take the last timePortion days from the data
  and they will be used to predict the next day stock price
*/
export const generateNextDayPrediction = (data: number[], timePortion: number): number[] => {
  let size = data.length;
  let features = new Array<number>();

  for (let i = (size - timePortion); i < size; i++) {
    features.push(data[i]);
  }

  return features;
}

/*
  Scaling feature using min-max normalization.
  All values will be between 0 and 1
*/
export const minMaxScaler = (data: number[], min: number, max: number) => {
  const scaledData = data.map((value) => (value - min) / (max - min));

  return {
    data: scaledData,
    min,
    max
  }
}


/*
  Revert min-max normalization and get the real values
*/
export const minMaxInverseScaler = (data: Float32Array | Int32Array | Uint8Array | number[], min: number, max: number) => {
  const scaledData = data.map((value) => value * (max - min) + min);

  return {
    data: scaledData,
    min,
    max
  }
}


/*
  Get min/max value from array
*/
function minMax2DArray(arr: number[]) {
  let max = -Number.MAX_VALUE;
  let min = Number.MAX_VALUE;

  arr.forEach((e) => {
    if (max < e) {
      max = e;
    }
    if (min > e) {
      min = e;
    }
  });
  return { max, min };
}


/*
  Adds days to given date
*/
export function addDays(date: Date, days: number) {
  const result = new Date(date);
  result.setDate(result.getDate() + days);
  return result;
}