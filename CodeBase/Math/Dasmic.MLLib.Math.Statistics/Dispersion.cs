using System;
using System.Threading;
using System.Linq;
using System.Threading.Tasks;
using Dasmic.MLLib.Common.Exceptions;


namespace Dasmic.MLLib.Math.Statistics
{
    public class Dispersion
    {
        /// <summary>
        /// Gets the population variance of a series
        /// </summary>
        /// <param name="values"></param>
        /// <returns>Variance of the series</returns>
        public static double VariancePopulation(double [] values)
        {
            double var=0;
            double mean = Mean(values);
            
            foreach (double value in values)        
            {
                //Both opearations need lock, hence cant be parallelized               
                var += value*value;
            }

            //Apply simple formula
            var = var / values.Count() - (mean*mean);
            return var;
        }

        /// <summary>
        /// Gets the population variance of a series
        /// if mean is known. This avoids double computation
        /// of the mean
        /// </summary>
        /// <param name="values"></param>
        /// <returns>Variance of the series</returns>
        public static double VariancePopulation(double[] values, double mean)
        {
            double var = 0;
            
            foreach (double value in values)
            {
                //Both opearations need lock, hence cant be parallelized
                var += value * value;
            }

            var = var / values.Count() - (mean * mean);
            
            return var;
        }



        /// <summary>
        /// Get the sample variance of a series
        /// </summary>
        /// <param name="values"></param>
        /// <returns>Variance of the series</returns>
        public static double VarianceSample(double[] values)
        {
            double var = 0;
            double mean = Mean(values);
            object obj = new object();
            Parallel.ForEach(values, (value) =>
            {
                double tmpVal;
                tmpVal = mean - value;
                lock (obj)
                { 
                    var += tmpVal * tmpVal;
                }
            });

            var = var / (values.Count()-1); //Sample variance is n-1
            return var;
        }

        public static double StandardDeviationSample(double [] values)
        {
            return System.Math.Sqrt(VarianceSample (values));
        }

        public static double StandardDeviationPopulation(double[] values)
        {
            return System.Math.Sqrt(VariancePopulation(values));
        }

        public static double StandardDeviationPopulation(
                            double[] values,double mean)
        {
            return System.Math.Sqrt(VariancePopulation(values,mean));
        }

        /// <summary>
        /// Mean of vector
        /// </summary>
        /// <param name="values"></param>
        /// <returns></returns>
        public static double Mean(double[] values)
        {
            return values.Average();
        }

        /// <summary>
        /// Get sum of vector
        /// </summary>
        /// <param name="values"></param>
        /// <returns></returns>
        public static double Sum(double[] values)
        {
            return values.Sum();
        }

        /// <summary>
        /// Mode of vector
        /// </summary>
        /// <param name="values"></param>
        /// <returns></returns>
        public static double Mode(double[] values)
        {
            double mode = Double.NaN;
            var groups = values.GroupBy(v => v);
            int maxCount = groups.Max(g => g.Count());

            if (maxCount == 1 && values.Length > 1) return mode; //No repeat numbers

            mode = groups.First(g => g.Count() == maxCount).Key;
            return mode;
        }


        public static double Median(double[] values)
        {
            double median=0;
            var orderedList= values.OrderBy(v => v).ToArray<double>();
            if(orderedList.Count()%2 == 0)//Even
            {
                int idx1 = orderedList.Count() /2;
                int idx2 = idx1+1;
                median = (orderedList[idx1-1] + orderedList[idx2-1]) / 2;
            }
            else
            {
                int idx1 = (orderedList.Count()+1) / 2;
                median = orderedList[idx1-1]; //Adjust for base 0 ordering
            }

            return median;
        }

        
        public static double Range(double[] values)
        {
            double range=0;
            range = values.Max() - values.Min();
            return range;
        }

        /// <summary>
        /// values1 and values2 should be same sized vectors
        /// sum(xy)/sqrt(sum(x^2) * sum(y^2)
        /// </summary>
        /// <param name="value1"></param>
        /// <param name="value2"></param>
        /// <returns></returns>
        public static double CorrelationPearson(double []values1,
                                    double[] values2)
        {
            if (values1.Length != values2.Length)
                throw new ArgumentException();

            //Compute sum of multiplication
            double xy=0,x2=0,y2=0,x=0,y=0;
            int N = values1.Length;
            //Parallel.For(0, values1.Length, (idx) =>
            //No need for a parallel loop as each line will have to interlocked
            for (int idx=0;idx<N;idx++)
            {
                x += values1[idx];
                y += values2[idx];
                xy += (values1[idx] * values2[idx]);
                x2 += System.Math.Pow(values1[idx], 2);
                y2 += System.Math.Pow(values2[idx], 2);
            };//);

            return (N *xy - (x*y)) /
                    System.Math.Sqrt(((N * x2) - System.Math.Pow(x,2)) *
                                ((N * y2) - System.Math.Pow(y, 2)));
        }


        /// <summary>
        /// Compute the Sample Covariance Matrix of data
        /// Data is passed as a 2D array with columns in first index and rows in second
        /// 
        /// Returns CoVariance matrix of size N x N where n is number of dimensions
        /// </summary>
        /// <param name="data"></param>
        /// <param name="maxParallelThreads"></param>
        /// <param name="isDataZeroCenter">Use if data is substracted from mean like in PCA</param>
        /// <returns></returns>
        public static double[][] CovarianceMatrixSample(double [][] data,
                                                    int maxParallelThreads=-1,
                                                    bool isDataZeroCenter=false)
        {
            double[][] coVarMatrix = new double[data.Length][];            
            double[] mean = new double[data.Length];
            //Compute mean and Init coVar
            Parallel.For(0, data.Length, new ParallelOptions { MaxDegreeOfParallelism = maxParallelThreads },(col) =>
            {
                if (!isDataZeroCenter)
                    mean[col] = Mean(data[col]);
                else
                    mean[col] = 0;
                coVarMatrix[col] = new double[data.Length];
            });
            
            for (int col=0;col<data.Length;col++)
            {
                for (int row = 0; row < data.Length; row++)
                {
                    coVarMatrix[col][row] = CoVarianceSample(data[col], data[row], 
                        mean[col], mean[row]);
                }
            }

            return coVarMatrix;
        }

        /// <summary>
        /// Makes data for every column data centered
        /// </summary>
        /// <param name="data"></param>
        /// <param name="ignoreLastColumn">Do not use last column in return data</param>
        /// <param name="maxParallelThreads"></param>
        /// <returns></returns>
        public static double[][] GetZeroCenteredData(double [][] data,
                                                     bool ignoreLastColumn=false,
                                                     int maxParallelThreads=-1)
        {
            double[][] newData;
            if (ignoreLastColumn)
                newData = new double[data.Length - 1][];
            else
                newData = new double[data.Length][];

            Parallel.For(0, newData.Length,
                           new ParallelOptions { MaxDegreeOfParallelism = maxParallelThreads },
                           col =>
                           //for (int col=0;col<newData.Length;col++)
            {
                               //if (col != indexTargetAttribute)
                               newData[col] = data[col];
                               double mean = Dispersion.Mean(newData[col]);

                               for (int row = 0; row < newData[col].Length; row++)
                               { //This will make each column's mean as 0
                                   newData[col][row] = newData[col][row] - mean;
                               }
            });

            return newData;
        }

        /// <summary>
        /// Computes the CoVariance between 2 data types
        /// 
        /// Both data types should have same number of data points
        /// </summary>
        /// <param name="data1"></param>
        /// <param name="data2"></param>
        /// <returns></returns>
        public static double CoVarianceSample(double[] data1, 
                                            double[] data2,
                                            double mean1=double.NaN,
                                            double mean2 = double.NaN)
        {
            if (data1.Length != data2.Length)
                throw new InvalidDataException();

            if(mean1.Equals(double.NaN))
                mean1 = Mean(data1);
            if (mean2.Equals(double.NaN))
                mean2 = Mean(data2);            

            double coVar = 0;
            for (int row = 0; row < data1.Length; row++)
            {
                coVar += (data1[row] - mean1) * (data2[row] - mean2);
            }
            coVar = coVar / (data1.Length-1);
            return coVar;
        }

    }
}
