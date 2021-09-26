import { readFile } from "fs/promises";
import { Data, LoadData, convertData } from "./helpers.js";

export type Symbol = "BTCUSDT" | "ETHUSDT" | "BNBUSDT" | "EGLDUSDT" | "SOLUSDT";
export type TimeFrame = "1min" | "15m" | "1h" | "4h" | "1d" | "1w" | "1M";

export const getData = async (symbol: Symbol, tf: TimeFrame): Promise<Data[]> => {
  const loadData: LoadData[] = JSON.parse(
    await readFile(`./data/1632380348209/${symbol}-${tf}.json`, "utf-8")
  );
  const data = loadData.map(convertData);
  return data;
};
