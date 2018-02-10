using System;
using System.Threading.Tasks;
using Dasmic.MLLib.Common.Exceptions;
using Dasmic.Portable.Core;

namespace Dasmic.MLLib.Math.Matrix
{
    public partial class MatrixOperations
    {              
        
        /// <summary>
        /// Computed QR of a Matrix using Given's rotation
        ///        
        /// </summary>
        /// <param name="matrix">Input Matrix</param>
        /// <param name="Q"></param>
        /// <param name="R"></param>
        /// <returns>Matrix in QR Form</returns>
        public void QRDecomposition_Givens_Rotation(
                                double[][] matrix,                                
                                ref double[][] Q,
                                ref double[][] R,                                
                                bool shouldVerify=true)
        {
            double [][] tmpQ = null;
            if (shouldVerify)
            {
                GeneralVerifyAndIfSquare(matrix);
            }

            //Start Given's rotation
            //Example from: https://en.wikipedia.org/wiki/Givens_rotation                        
            //Apply Given's rotation from left

            //Parallel.For(0, hMatrix.Length,
            //   new ParallelOptions { MaxDegreeOfParallelism = MaxParallelThreads },
            //   col =>
          

            double[][][] G = new double[matrix.Length][][];
            //R = SupportFunctions.Get2DArray(matrix.Length, matrix.Length,-1);

            for (int col=0;col<matrix.Length-1;col++) //Parallelize this
            {                
                for (int row = col + 1; row < matrix.Length; row++)
                {
                    //Will 0 element matrix[col][row]
                    if (!SupportFunctions.DoubleCompare(matrix[col][row], 0))
                    { //If element is already 0 do not do non required computation
                        G[col] = IdentityMatrix(matrix.Length);
                        //Compute r
                        double tempR = System.Math.Pow(matrix[col][col], 2) +
                                    System.Math.Pow(matrix[col][row], 2);
                        tempR = System.Math.Sqrt(tempR);
                        double tempC = matrix[col][col] / tempR;
                        double tempS = -matrix[col][row] / tempR;
                        G[col][col][col] = tempC;
                        G[col][row][row] = tempC;
                        G[col][row][col] = -tempS;
                        G[col][col][row] = tempS;

                        matrix = Multiply(G[col], matrix); //Do not use original matrix                 

                        //Only old matrix is used
                        //R[col][col] = newMatrix[col][col];
                        //R[row][row] = newMatrix[row][row];
                        //R[row][col] = newMatrix[row][col];
                        //R[col][row] = newMatrix[col][row];

                        if (tmpQ == null)
                            tmpQ = Transpose(G[col]);
                        else
                            tmpQ = Multiply(tmpQ, Transpose(G[col]));
                    }
                }


            }//);
              
            Q = tmpQ;
            R = matrix;  
            
        } //QRDecomposition_Givens
    }
}
