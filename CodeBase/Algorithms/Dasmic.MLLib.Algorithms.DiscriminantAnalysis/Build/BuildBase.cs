using System;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using Dasmic.MLLib.Common.MLCore;
using Dasmic.MLLib.Common.Exceptions;
using System.Collections.Generic;
using Dasmic.MLLib.Math.Statistics;
using Dasmic.MLLib.Math.Matrix;

namespace Dasmic.MLLib.Algorithms.DiscriminantAnalysis
{
    public abstract class BuildBase : Common.MLCore.BuildBase
    {
        protected MatrixOperations _mo;

        public BuildBase()
        {
            _mo = new MatrixOperations();
        }

        public override abstract Common.MLCore.ModelBase BuildModel(double[][] trainingData,
                     string[] attributeHeaders,
                     int indexTargetAttribute);

        
        /// <summary>
        /// Returns the class probabilities of each group
        /// 
        /// </summary>
        /// <param name="classBasedInputMatrix"></param>
        /// <param name="trainingData"></param>
        /// <returns></returns>
        protected double[] GetClassProbabilities(
                                            List<double[][]> classBasedInputMatrix)
        {
            //Assume: Data is validated
            double[] groupProbabilities = new double[classBasedInputMatrix.Count ];
            double totalElements = _trainingData[0].Length;

            for(int ii= 0;ii<classBasedInputMatrix.Count;ii++)
            {
                groupProbabilities[ii] = (double)classBasedInputMatrix[ii][0].Length /
                    totalElements;
            }

            return groupProbabilities;
        }


        //Return mean matrix of dataset
        //Private members are not being used due to unit test purposes
        /// <summary>
        /// Assume target attributes is last column in Training Data
        /// </summary>
        /// <param name="trainingData"></param>
        /// <param name="indexTargetAttribute"></param>
        /// <param name="noOfAttributes"></param>
        /// <returns></returns>
        protected double[] GetDataSetMeanMatrix()
        {
            double[] mean = new double[_noOfAttributes];
            //int idx = 0;
            //Assumption: Target Attributer is last column
            Parallel.For(0, _trainingData.Length-1, colIdx => //For each column
            {
                //if (colIdx != indexTargetAttribute)
                //{
                    mean[colIdx] = Dispersion.Mean(_trainingData[colIdx]);                    
               //}
            });           
            return mean;
        }

        /// <summary>
        /// Returns the corrected class mean matrix
        /// 
        /// This is the class mean matrix substracted by global mean matrix
        /// 
        /// Matrices are not of same size
        /// </summary>
        /// <param name="classMeanMatrix"></param>
        /// <param name="globalMeanMatrix"></param>
        /// <returns></returns>
        protected List<double[][]> GetCorrectedDataMatrix(
                                    List<double[][]> classMeanMatrix,
                                    double[] globalMeanMatrix)
        {
            List<double[][]> correctedMatrix = new List<double[][]>();

            // Column in globalMeanMatrix should match those 
            // in classMeanMatrix
            foreach (double[][] cmm in classMeanMatrix)
            {
                //ccMatrix: Class corrected  matrix
                double[][] ccMatrix = new double[globalMeanMatrix.Length][];
                //For each column
                for (int colIdx = 0;
                    colIdx < globalMeanMatrix.Length;
                    colIdx++)
                {
                    ccMatrix[colIdx] = new double[cmm[colIdx].Length];
                    for (int rowIdx = 0;
                            rowIdx < ccMatrix[colIdx].Length; rowIdx++)
                    {
                        ccMatrix[colIdx][rowIdx] = cmm[colIdx][rowIdx] - 
                            globalMeanMatrix[colIdx];
                    }
                }
                correctedMatrix.Add(ccMatrix);
            }

            return correctedMatrix;
        }


        /// <summary>
        /// Builds a covariance matrix based on corrected Matrix
        /// </summary>
        /// <param name="classCorrectedMatrix"></param>
        /// <returns></returns>
        protected List<double[][]> GetCoVarianceMatrix(
                                    List<double[][]> classCorrectedMatrix)
        {
            List<double[][]> covarianceMatrix = new List<double[][]>();
            //Add dummy items
            Parallel.For(0, classCorrectedMatrix.Count, idx =>
            {
                covarianceMatrix.Add(null);
            });

           

            //TODO: 
            //Parallelize this op
            //foreach (double[][] ccMatrix in classCorrectedMatrix)
            var oLock = new Object();
            Parallel.For(0, classCorrectedMatrix.Count, idx =>
              {
                    double[][] ccMatrix = classCorrectedMatrix[idx];
                    double[][] ccmTranspose = _mo.Transpose(ccMatrix);
                    double[][] ccvMatrix = _mo.Multiply(ccmTranspose, ccMatrix);
                    //Divide by number of rows
                    ccvMatrix = _mo.DivideByScalar(ccvMatrix, ccMatrix[0].Length);
                  
                    lock (oLock) //List ops are not thread safe
                    {
                      covarianceMatrix.RemoveAt(idx);
                      covarianceMatrix.Insert(idx, ccvMatrix);
                    }
              });

            return covarianceMatrix;
        }

        
        protected double[][] GetPooledCoVarianceMatrix(
                                    List<double[][]> covarianceMatrix,
                                    double[] classProbabilities)
        {
            if (covarianceMatrix.Count !=
                    classProbabilities.Length)
                throw new AttributesCountMismatchException();
            
            //Assign Matrix
            double[][] pooledcvMatrix = new double[covarianceMatrix[0].Length][];
            //for(int ii=0;ii< pooledcvMatrix.Length; ii++)
            Parallel.For(0, pooledcvMatrix.Length, ii =>
            {
                pooledcvMatrix[ii] = new double[covarianceMatrix[0][ii].Length];
            });

            //for (int row = 0; row < pooledcvMatrix.Length; row++)
            Parallel.For(0, pooledcvMatrix.Length, row =>
            {
                for (int col = 0; col < pooledcvMatrix[0].Length; col++)
                {
                    //for (int col1 = 0; col1 < pooledcvMatrix[0].Length; col1++)
                    for (int col1 = 0; col1 < classProbabilities.Length; col1++)
                        pooledcvMatrix[col][row] +=
                            classProbabilities[col1] * covarianceMatrix[col1][col][row];
                }
            });
            return pooledcvMatrix;            
        }



    }
}
