using System;
using System.Threading.Tasks;
using System.Collections.Concurrent;
using Dasmic.MLLib.Common.Exceptions;

namespace Dasmic.MLLib.Math.Matrix
{
    public partial class MatrixOperations
    {
        /// <summary>
        /// Computes the Adjucate of a square matrix using LU method
        /// </summary>
        /// <param name="matrix"></param>
        /// <returns></returns>
        public double Adjucate(double[][]
                                matrix)
        {
           
            VerifyMatrix(matrix);

            //Check if square matrix
            if (matrix.Length != matrix[0].Length)
                throw new InvalidMatrixException();

            //Get L and U Matrices
            return 0;
        }
    }
}
