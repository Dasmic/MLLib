using System;
using System.Threading.Tasks;
using System.Collections.Concurrent;
using Dasmic.MLLib.Common.Exceptions;

namespace Dasmic.MLLib.Math.Matrix
{
    public partial class MatrixOperations
    {
        //This is for a general matrix
        //Row/Col concepts can differ from those used in trainingData
        public double[][] Transpose(double[][]
                                matrix)
        {
            VerifyMatrix(matrix);

            double[][] tMatrix = new double[matrix[0].Length][];
            Parallel.For(0, tMatrix.Length, ii =>
              {
                  tMatrix[ii] = new double[matrix.Length];
              });

            //Start Transpose Op
            //Row here is used for clarity to denote row size of tMatrix, 
            //it may differ from actual row
            //same applies to col
            //for (int row = 0; row < tMatrix.Length; row++)
            Parallel.For(0, tMatrix.Length, row =>
            {
                for (int col = 0; col < tMatrix[0].Length; col++)
                    tMatrix[row][col] = matrix[col][row];
            });

            return tMatrix;
        }        
    }
}
