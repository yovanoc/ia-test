import { readFile } from "fs/promises";
import { Data, LoadData, convertData } from "./helpers.js";

export const getData = async (symbol: string): Promise<Data[]> => {
  const loadData: LoadData[] = JSON.parse(await readFile(`./data/1626626949735/${symbol}-1d.json`, "utf-8"));
  const data = loadData.map(convertData);
  return data;
}