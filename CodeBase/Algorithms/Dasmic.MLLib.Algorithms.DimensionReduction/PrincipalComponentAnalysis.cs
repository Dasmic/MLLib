using System;
using System.Collections.Generic;
using System.Linq;
using Dasmic.MLLib.Math.Matrix;
using Dasmic.MLLib.Math.Statistics;
using Dasmic.MLLib.Common.Exceptions;

namespace Dasmic.MLLib.Algorithms.DimensionReduction
{
    public class PrincipalComponentAnalysis
    {
        public int Rank { get; set; }
        public double Tolerance { get; set; }

        public PrincipalComponentAnalysis()
        {
            Tolerance = 0;
            Rank = int.MaxValue;
        }

        public double[][] GetPrincipleFeatures(double[][] trainingData,
                             string[] attributeHeaders,                             
                             ref double[] standardDeviations,
                             ref double[][] rotationMatrix,
                             int maxParallelThreads=-1, 
                             int indexTargetAttribute = -1)
        {
            if (trainingData.Length !=
                        attributeHeaders.Length)
                throw new AttributesCountMismatchException();

            bool ignoreLastColumn = indexTargetAttribute >= 0?true:false;
            double[][] zeroCenteredData = Dispersion.GetZeroCenteredData(trainingData,
                                                                ignoreLastColumn,
                                                                maxParallelThreads);
            if (Rank > zeroCenteredData.Length)
                Rank = zeroCenteredData.Length;

            //Generate Covariance Matrix from NewData
            double[][] coVarMatrix = Dispersion.CovarianceMatrixSample(zeroCenteredData, maxParallelThreads,true);

            //Compute EigenValue/Vectors of coVarMatrix
            double[][] eigenVectors = null;
            double[] eigenValues = null;

            MatrixOperations mo = new MatrixOperations();

            mo.EigenValues(coVarMatrix, ref eigenValues,
                                  ref eigenVectors);

            //Form Feature Vector
            //Sort indexes by EigenValues
            //List<int> A = new List<int>() { 3, 2, 1 };

            var sorted = eigenValues
                .Select((x, i) => new KeyValuePair<double, int>(x, i))
                .OrderByDescending(x => x.Key)
                .ToList();

            //List<int> B = sorted.Select(x => x.Key).ToList();
            List<int> idxSorted = sorted.Select(x => x.Value).ToList();

            //Convert SD Matrix
            List<double> standardDevList = new List<double>();
            for (int idx = 0; idx < Rank; idx++)
            {
                if (eigenValues[idxSorted[idx]] >= 0)
                {
                    double sd = System.Math.Sqrt(eigenValues[idxSorted[idx]]);
                    if (sd >= Tolerance)
                        standardDevList.Add(sd);
                }            
            }

            standardDeviations = standardDevList.ToArray();

            //EigenValues are already sorted in decreasing order
            //Hence idxSorted[0] already has highest eigenvalue/std dev
            rotationMatrix = new double[standardDeviations.Length][];
            
            for (int idx=0;idx< standardDeviations.Length; idx++)
            {                
                rotationMatrix[idx] = eigenVectors[idxSorted[idx]];
            }

            //Derive New Data Set
            double[][] finalPCAData = mo.Multiply(mo.Transpose(rotationMatrix),
                                                   mo.Transpose(zeroCenteredData));
            finalPCAData = mo.Transpose(finalPCAData);
            return finalPCAData;
        }
    }
}
