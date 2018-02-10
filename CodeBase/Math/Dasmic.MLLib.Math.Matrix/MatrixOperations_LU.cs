using System;
using System.Threading.Tasks;
using Dasmic.MLLib.Common.Exceptions;

namespace Dasmic.MLLib.Math.Matrix
{
    public partial class MatrixOperations
    {
        /// <summary>
        /// Returns the Permutation Matrix of a given Matrix
        /// 
        /// Exchange rows of original matrix such that the diagonal 
        /// elements are the biggest number in its own column
        /// 
        /// Ref: https://epxx.co/artigos/ludecomp.html
        /// </summary>
        /// <param name="matrix"></param>
        public double[][] PermutationMatrix(double[][] 
                                        matrix)
        {
            double[][] pMatrix = IdentityMatrix(matrix.Length);

            int colIdx = 0,rowIdx = 0,cIdx=0;
            double tmp = 0;
            //Should not be parallelized as it is a sequential op.
            for (colIdx = 0; colIdx < matrix.Length ; colIdx++)
            {
                int row = colIdx;
                double val = 0;
                for (rowIdx = colIdx; 
                    rowIdx < matrix.Length; rowIdx++)
                {
                    tmp = System.Math.Abs(matrix[colIdx][rowIdx]);
                    if (val < tmp)
                    {
                        val = tmp;
                        row = rowIdx;
                    }
                }

                if (colIdx != row)
                {
                   //Swap the rows: row and colIdx
                    for (cIdx = 0; cIdx < matrix.Length; ++cIdx)
                    {
                        tmp = pMatrix[cIdx][colIdx];
                        pMatrix[cIdx][colIdx] = pMatrix[cIdx][row];
                        pMatrix[cIdx][row] = tmp;
                    }
                }
            }
            return pMatrix;
        }
        
        
        
        /// <summary>
        /// LU decomposition of Matrix using Crout's method
        /// </summary>
        /// <param name="matrix"></param>
        /// <param name="L"></param>
        /// <param name="U"></param>
        public void LupMatrix(double[][] matrix,
                                ref double[][] L,
                                ref double[][] U,
                                ref double[][] P)
        {
            LupMatrix_Crout(matrix, ref L, 
                    ref U, ref P, true);
        }


        /// <summary>
        /// LU decomposition of Matrix using Crout's method
        /// 
        /// Private method, do not verify matrix again
        /// Assume Matrix is square
        /// 
        /// </summary>
        /// <param name="matrix"></param>
        /// <param name="L"></param>
        /// <param name="U"></param>
        /// <param name="P"></param>
        /// <returns></returns>
        private void LupMatrix_Crout(double[][]
                                matrix, ref double[][] L,
                                ref double[][] U,
                                ref double[][] P,
                                bool shouldVerify)
        {
            if (shouldVerify)
            {
                VerifyMatrix(matrix);
                //Check if square matrix
                if (matrix.Length != matrix[0].Length)
                    throw new InvalidMatrixException();
            }

            
            int size = matrix.Length;

            double [][] tmpL = GetNewMatrix(matrix);
            double[][] tmpU = GetNewMatrix(matrix);
            P = PermutationMatrix(matrix);
            double[][] tmpMatrix = Multiply(P, matrix);
            
            //Init values for U
            Parallel.For(0, size, ii =>
            {
               tmpU[ii][ii] = 1;
            });

            //Do not parallelize since both tmpL[col][row] and tmpL[row][col] are used
            for (int col = 0; col < size; col++)
            // Parallel.For(0, size, col =>
            {
                  double sum = 0;
                  for (int row = col; row < size; row++)
                  {
                      sum = 0;
                      for (int idx = 0; idx < col; idx++)
                      {
                          sum = sum + tmpL[idx][row] * 
                                    tmpU[col][idx];
                      }
                      tmpL[col][row] = tmpMatrix[col][row] -
                                          sum;
                  }

                  for (int row = col; row < size; row++)
                  {
                      sum = 0;
                      for (int idx = 0; idx < col; idx++)
                      {
                          sum = sum + tmpL[idx][col] *
                                          tmpU[row][idx];
                      }
                      if (tmpL[col][col] == 0) //Prevent division by 0 exception
                          throw new LUDecompositionException();
                      else
                          tmpU[row][col] = (tmpMatrix[row][col] - sum) /
                                          tmpL[col][col];
                  }
              }//);

            U = tmpU;
            L = tmpL;
        }
    }
}
