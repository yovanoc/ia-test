import { ChartConfiguration, ChartTypeRegistry } from 'chart.js';
import { ChartJSNodeCanvas } from 'chartjs-node-canvas';
import { writeFile } from "fs/promises";

// Re-use one service, or as many as you need for different canvas size requirements
const smallChartJSNodeCanvas = new ChartJSNodeCanvas({ width: 400, height: 400, type: "svg" });
const mediumCChartJSNodeCanvas = new ChartJSNodeCanvas({ width: 1920, height: 1080, type: "svg"  });
const bigCChartJSNodeCanvas = new ChartJSNodeCanvas({ width: 10666, height: 6000, type: "svg"  });

const render = {
  smallChart: (configuration: ChartConfiguration<keyof ChartTypeRegistry>) => smallChartJSNodeCanvas.renderToBufferSync(configuration),
  mediumChart: (configuration: ChartConfiguration<keyof ChartTypeRegistry>) => mediumCChartJSNodeCanvas.renderToBufferSync(configuration),
  bigChart: (configuration: ChartConfiguration<keyof ChartTypeRegistry>) => bigCChartJSNodeCanvas.renderToBufferSync(configuration)
};

type Size = "small" | "medium" | "big";

const sizes: Size[] = ["small", "medium", "big"];

/*
    Plot two arrays of data using Chart.js lib
*/
export const saveChart = async (symbol: string, data1: number[], data2: number[], labels: string[]) => {

  const config: ChartConfiguration<keyof ChartTypeRegistry> = {
    type: "line",
    data: {
      labels,
      datasets: [{
        label: 'Predicted',
        fill: false,
        backgroundColor: 'red',
        borderColor: 'red',
        data: data2,
      }, {
        label: 'Actual',
        backgroundColor: 'blue',
        borderColor: 'blue',
        data: data1,
        fill: false,
      }]
    },
    options: {
      responsive: true,
      plugins: {
        title: {
          display: true,
          text: 'Stock Price Prediction'
        },
        tooltip: {
          mode: "index",
          intersect: false,
        },
      },
      hover: {
        mode: 'nearest',
        intersect: true
      },
      scales: {
        x: {
          display: true,
          title: {
            display: true,
            text: 'Date'
          }
        },
        y: {
          display: true,
          title: {
            display: true,
            text: 'Stock Value'
          }
        }
      }
    }
  };

  const renderSize = async (size: Size) => {
    const buffer = render[`${size}Chart`](config);
    await writeFile(`./results/${symbol}-${size}.svg`, buffer);
  }

  await Promise.all(sizes.map(renderSize));
}
