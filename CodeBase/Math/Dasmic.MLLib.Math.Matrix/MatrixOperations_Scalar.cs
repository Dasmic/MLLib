using System.Threading.Tasks;
using System;
using System.Collections.Concurrent;
using Dasmic.MLLib.Common.Exceptions;

namespace Dasmic.MLLib.Math.Matrix
{
    public partial class MatrixOperations
    {
        #region Scalar Operations
        /// <summary>
        /// Divides the Matrix by a scalar values
        /// </summary>
        /// <param name="matrix"></param>
        /// <param name="scalar"></param>
        /// <returns></returns>
        public double[][] DivideByScalar(double[][]
                               matrix, double scalar)
        {
            VerifyMatrix(matrix);
            double[][] newMatrix =
                GetNewMatrix(matrix);

            //for (col = 0; col < newMatrix.Length; col++)
            Parallel.For(0, newMatrix.Length, col =>
            {
                int row = 0;
                for (row = 0; row < newMatrix[0].Length; row++)
                    newMatrix[col][row] = matrix[col][row] / scalar;
            });
            return newMatrix;
        }

        /// <summary>
        /// Multiplies the Matrix by a scalar values
        /// </summary>
        /// <param name="matrix"></param>
        /// <param name="scalar"></param>
        /// <returns></returns>
        public double[][] MultiplyByScalar(double[][]
                               matrix, double scalar)
        {
            double[][] newMatrix =
                GetNewMatrix(matrix);

            int row = 0;
            //for (col = 0; col < newMatrix.Length; col++)
            Parallel.For(0, newMatrix.Length, col =>
            {
                for (row = 0; row < newMatrix[0].Length; row++)
                    newMatrix[col][row] = matrix[col][row] * scalar;
            });
            return newMatrix;
        }
        #endregion
    }
}
