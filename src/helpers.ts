import { TimeFrame } from "./data.js";

export type Data = {
  openTime: Date;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  closeTime: Date;
  average: number;
}

export type LoadData = {
  openTime: string;
  open: string;
  high: string;
  low: string;
  close: string;
  volume: string;
  closeTime: string;
  average: string;
}

export const convertData = (data: LoadData): Data => ({
  openTime: new Date(data.openTime),
  closeTime: new Date(data.closeTime),
  open: parseFloat(data.open),
  high: parseFloat(data.high),
  low: parseFloat(data.low),
  close: parseFloat(data.close),
  volume: parseFloat(data.volume),
  average: parseFloat(data.average),
});

/*
    Process the price data
    Create the train features and labels for cnn
    Each prediction is base on previous timePortion days
    ex. timePortion=7, prediction for the next day is based to values of the previous 7 days
*/
export const processData = (data: Data[], timePortion: number, splitDate: Date) => {
  const trainX = new Array<number>();
  const trainY = new Array<number>();
  const visX = new Array<number>();
  const visY = new Array<number>();
  const size = data.length;

  const features = new Array<number>();
  for (let i = 0; i < size; i++) {
    features.push(data[i].close);
  }
  const featuresDates = new Array<Date>();
  for (let i = 0; i < size; i++) {
    featuresDates.push(data[i].closeTime);
  }

  // Scale the values
  const { min, max } = minMax2DArray(features);
  const scaledData = minMaxScaler(features, min, max);
  const scaledFeatures = scaledData.data;

  // Create the train sets
  for (let i = timePortion; i < size; i++) {

    const d = featuresDates[i];
    const arrY = d < splitDate ? trainY : visY;
    const arrX = d < splitDate ? trainX : visX;

    arrY.push(scaledFeatures[i]);

    for (let j = (i - timePortion); j < i; j++) {
      arrX.push(scaledFeatures[j]);
    }
  }

  return {
    trainSize: trainY.length,
    visSize: visY.length,
    timePortion,
    trainX,
    trainY,
    visX,
    visY,
    min: scaledData.min,
    max: scaledData.max,
    originalData: features
  }
};


/*
  This will take the last timePortion entries from the data
  and they will be used to predict the next stock price
*/
export const generateNextPrediction = (data: number[], timePortion: number): number[] => {
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

export function addMsToDate(date: Date, ms: number) {
  return new Date(date.getTime() + ms);
}

export function timeFrameToMS(timeFrame: TimeFrame) {
  switch (timeFrame) {
    case "1min":
      return ms.minutes(1);
    case "15m":
      return ms.minutes(15);
    case "1h":
      return ms.hours(1);
    case "4h":
      return ms.hours(4);
    case "1d":
      return ms.days(1);
    case "1w":
      return ms.weeks(1);
    case "1M":
      return ms.months(1);
  }
}

function calc(m: number) {
  return function (n: number) {
    return Math.round(n * m);
  };
}

export const ms = {
  seconds: calc(1e3),
  minutes: calc(6e4),
  hours: calc(36e5),
  days: calc(864e5),
  weeks: calc(6048e5),
  months: calc(26298e5),
  years: calc(315576e5)
} as const;