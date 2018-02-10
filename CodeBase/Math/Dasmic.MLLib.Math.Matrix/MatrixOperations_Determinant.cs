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
        /// Computes the Determinant of a square matrix using LUP method
        /// 
        /// If Determinant is 0 this method will raise an exception
        /// </summary>
        /// <param name="matrix"></param>
        /// <returns></returns>
        public double Determinant(double[][]
                                matrix)
        {
            double det = 0;
            VerifyMatrix(matrix);

            //Check if square matrix
            if (matrix.Length != matrix[0].Length)
                throw new InvalidMatrixException();

            //Get L and U Matrices
            double[][] L = null, U = null,P=null;
            

            if (matrix.Length == 1)
                det = matrix[0][0];
            else 
            {
                LupMatrix(matrix, ref L, ref U, ref P);
                double detL=0, detU=0,detP=0;
                //Compute in parallel
                Parallel.Invoke(() =>
                    detL = DeterminantTriangular(L),
                    () => detU = DeterminantTriangular(U),
                    () => detP = DeterminantPermutation(P));
                det = detL * detU * detP;
            }
            
            return det;
        }

        /// <summary>
        /// Computes the Determinant of a triangular matrix
        /// 
        /// The determinant of triangular matrix is the product of diagonal matrix
        /// 
        /// This function does not check if matrix is triangular
        /// </summary>
        /// <param name="matrix"></param>
        /// <returns></returns>
        public double DeterminantTriangular(double[][]
                                matrix)
        {
            double det=1;
            
            //Check if square matrix
            if (matrix.Length !=
                        matrix[0].Length)
                throw new InvalidMatrixException();

            for(int row=0;row<matrix.Length;row++)
                det = det*matrix[row][row];
            
            return det;
        }

        /// <summary>
        /// Computes the determinant of Permutation Matrix
        /// 
        /// Determinant of such a matrix is -1^re where
        /// re is number of row exchanges
        /// 
        /// </summary>
        /// <param name="matrix"></param>
        /// <returns></returns>
        public double DeterminantPermutation(double[][]
                                matrix)
        {
            double det = 0;
            int count=0;

            Parallel.For(0, matrix.Length, rowIdx =>
            {
                 if (matrix[rowIdx][rowIdx] != 1)
                     Interlocked.Increment(ref count); 
                 //Add thread safe
            });

            det = System.Math.Pow(-1, System.Math.Ceiling(count / 2.0));
            return det;
        } 

    }
}
