using System;
using System.Threading;
using System.Threading.Tasks;
using System.Collections.Concurrent;
using Dasmic.MLLib.Common.Exceptions;

namespace Dasmic.MLLib.Math.Matrix
{
    public partial class MatrixOperations
    {
        /// <summary>
        /// Computes the cofactor matrix of a square matrix
        /// </summary>
        /// <param name="matrix"></param>
        /// <returns></returns>
        public double[][] CofactorMatrix(double[][]
                                matrix)
        {
            VerifyMatrix(matrix);

            //Check if square matrix
            if (matrix.Length != matrix[0].Length)
                throw new InvalidMatrixException();

            double[][] newMatrix = GetNewMatrix(matrix);

           
            for (int col = 0; col < matrix.Length; col++)
                for (int row = 0; row < matrix[0].Length; row++)
                {
                        double[][] tmpMatrix = GetReducedMatrix(matrix, col, row);
                        //compute minor
                        newMatrix[col][row] = Determinant(tmpMatrix);
                        //compute cofactor
                        newMatrix[col][row] = System.Math.Pow(-1, col + row) *
                                                    newMatrix[col][row];
                }
            
            return newMatrix;
        }

        /// <summary>
        /// Computes a new matrix that has the original values
        /// in col and row not present
        /// </summary>
        /// <param name="matrix"></param>
        /// <param name="col"></param>
        /// <param name="row"></param>
        /// <returns></returns>
        private double[][] GetReducedMatrix(double[][] matrix, 
                                            int col, int row)
        {
            double[][] tMatrix =
                new double[matrix.Length-1][];
            
            Parallel.For(0, tMatrix.Length, idx =>
            {
                tMatrix[idx] = new double[tMatrix.Length];
            });

            int colIdx=0,rowIdx=0;
            Parallel.For(0, tMatrix.Length, c =>
            {
                if (colIdx == col) //If same as col
                    Interlocked.Increment(ref colIdx); //Move to next

                //Start for Row
                rowIdx = 0;
                for (int r = 0; r < tMatrix[0].Length; r++)
                {
                    if (rowIdx == row)//If not same as col
                        Interlocked.Increment(ref rowIdx);
                    tMatrix[c][r] =
                            matrix[colIdx][rowIdx];
                    Interlocked.Increment(ref rowIdx);
                }
                Interlocked.Increment(ref colIdx);   
            });
            return tMatrix;
        }
    }
}
