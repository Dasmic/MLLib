using System;
using System.Threading.Tasks;
using System.Collections.Concurrent;
using Dasmic.MLLib.Common.Exceptions;
using Dasmic.Portable.Core;

namespace Dasmic.MLLib.Math.Matrix
{
    public partial class MatrixOperations
    {
        public int MaxParallelThreads{get;set;}

        public MatrixOperations()
        {
            MaxParallelThreads = -1;
        }

        protected void GeneralVerifyAndIfSquare(double[][] matrix)
        {
            VerifyMatrix(matrix);
            if (!IsSquare(matrix))
                throw new InvalidMatrixException();
        }

        protected void VerifyMatrix(double[][] matrix)
        {
            if (matrix.Length == 0)
                throw new InvalidMatrixException();

            if (matrix[0].Length == 0)
                throw new InvalidMatrixException();

            // Use ConcurrentQueue to enable safe enqueueing from multiple threads.
            var exceptions = new ConcurrentQueue<Exception>();

            //foreach (double[] values in matrix)
            Parallel.ForEach(matrix, (values) =>
            {
               try
                {
                   if (values == null)
                       throw new InvalidMatrixException();
                   if (values.Length != matrix[0].Length)
                       throw new InvalidMatrixException();
                }
                catch (Exception ex)
                {
                    exceptions.Enqueue(ex);
                }
            });

            //Throw the first exception, if any were raised
            if(!exceptions.IsEmpty)
            { 
                Exception e;
                exceptions.TryDequeue(out e);
                throw e;
            }

        }

        /// <summary>
        /// Gets a new matrix same size as passed matrix
        /// </summary>
        /// <param name="matrix"></param>
        /// <returns></returns>
        protected double[][] GetNewMatrix(double[][]
                                matrix)
        {
            double[][] tMatrix =
                new double[matrix.Length][];

            //for (int idx = 0; idx < tMatrix.Length; idx++)
            Parallel.For(0, tMatrix.Length, idx =>
            {
                tMatrix[idx] = new double[matrix[0].Length];
            });
            return tMatrix;
        }

        
        /// <summary>
        /// Will copy two matrices. A new copy of copyToMatrix
        /// will be created
        /// </summary>
        /// <param name="fromMatrix"></param>
        /// <param name="copyToMatrix"></param>
        /// <param name="maxParallelThreads"></param>
        protected void CopyMatrix(double[][] fromMatrix, 
                                  ref double[][] copyToMatrix, 
                                  int maxParallelThreads = -1)
        {
            double [][] mNew = new double[fromMatrix.Length][];
            
            if (fromMatrix == null || mNew ==null )
                throw new InvalidMatrixException();
            if (fromMatrix.Length != mNew.Length)
                throw new InvalidMatrixException();
          

        Parallel.For(0, fromMatrix.Length,
         new ParallelOptions { MaxDegreeOfParallelism = maxParallelThreads },
        col =>
        //for (int col=0;col<fromMatrix.Length;col++)
         {
             mNew[col] = new double[fromMatrix.Length];
             Array.Copy(fromMatrix[col], mNew[col],mNew[col].Length);
         });

            copyToMatrix = mNew;
        }

        /// <summary>
        /// Compares 2 batrices and returns true if there are same
        /// </summary>
        /// <param name="matrix1"></param>
        /// <param name="matrix2"></param>
        /// <returns></returns>
        public bool CompareMatrix(double[][] matrix1,
                                     double[][] matrix2)
        {
            bool flag=true;

            if (matrix1.Length != matrix2.Length) return false;
            if (matrix1[0].Length != matrix2[0].Length) return false;

            for (int col = 0; col < matrix1.Length; col++)
                for (int row = 0; row < matrix1[0].Length; row++)
                    if(!SupportFunctions.DoubleCompare(matrix1[col][row], 
                                            matrix2[col][row]))
                    {
                        flag = false;
                        break;
                    }
            return flag;
        }

        /// <summary>
        /// Returns true if matrix is square
        /// </summary>
        /// <param name="matrix"></param>
        /// <returns></returns>
        protected bool IsSquare(double[][] matrix)
        {
            bool flag=false;
            if (matrix[0] != null)
                if (matrix.Length == matrix[0].Length)
                    flag = true;

            return flag;
        }

        /// <summary>
        /// Return true is matrix is Upper Triangular
        /// False otherwise. Upper triangular matrix has all
        /// elements below diagonal 0
        /// </summary>
        /// <param name="matrix"></param>
        /// <returns></returns>
        public bool IsUpperTriangular(double[][] matrix)
        {
            bool flag = true;
            for (int col = 0; col < matrix.Length-1; col++)
            {
                for (int row = col+1; row < matrix.Length; row++)
                {
                    if (!SupportFunctions.DoubleCompare(matrix[col][row], 0))
                    {
                        flag = false;
                        break;
                    }
                }
            }
            return flag;
        }

        /// <summary>
        /// Returns if a Matrix is a Diagonal, false otherwise
        /// 
        /// A Diagonal matrix has all elements 0 
        /// except ones on its diagonal
        /// 
        /// Note that the diagonal can have 0's
        /// 
        /// Use only for Square matrix
        /// </summary>
        /// <param name="matrix"></param>
        /// <returns></returns>
        public bool IsDiagonal(double[][] matrix)
        {
            if (!IsSquare(matrix)) return false;

            bool flag = true;
            for (int col = 0; col < matrix.Length; col++)
            {
                //Dont use this condition as diagonal can be zero                    
                for (int row = col + 1; row < matrix.Length; row++)
                {
                    if (!SupportFunctions.DoubleCompare(matrix[col][row], 0)
                        || !SupportFunctions.DoubleCompare(matrix[row][col], 0))
                    {                       
                        flag = false;
                        break;                        
                    }  
                }
            }
            return flag;
        }

    }
}
