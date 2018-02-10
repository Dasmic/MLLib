using System;
using System.Threading.Tasks;
using Dasmic.MLLib.Common.Exceptions;

namespace Dasmic.MLLib.Math.Matrix
{
    public partial class MatrixOperations
    {              
        /// <summary>
        /// QR decomposition of Matrix using Hessenberg Givens method
        /// </summary>
        /// <param name="matrix"></param>
        /// <param name="Q"></param>
        /// <param name="R"></param>
        /// <param name="U">Returned from Hessenberg</param>
        public void QRDecomposition_Hessenberg_Givens(double[][] matrix,                                                        
                                ref double[][] Q,
                                ref double[][] R,
                                ref double[][] U,
                                bool shouldVerify = true)
        {
            //Get Matrix in Hessenberg form
            if (shouldVerify)
            {
                GeneralVerifyAndIfSquare(matrix);
            }
            double[][] hMatrix = Hessenberg(matrix, ref U, false);
            QRDecomposition_Hessenberg(hMatrix, ref U, ref Q, ref R,false);
        }


        /// <summary>
        /// Computed QR of a given Hessenberg Matrix using Given's rotation
        /// 
        /// CAUTION: Matrix passed should be Hessenberg Matrix
        /// </summary>
        /// <param name="hMatrix">Hessenberg Matrix</param>
        /// <param name="Q"></param>
        /// <param name="R"></param>
        /// <returns>Matrix in QR Form</returns>
        public void QRDecomposition_Hessenberg(
                                double[][]hMatrix,
                                ref double[][] U,
                                ref double[][] Q,
                                ref double[][] R,                                
                                bool shouldVerify=true)
        {
            double [][] tmpQ = Q;
            if (shouldVerify)
            {
                GeneralVerifyAndIfSquare(hMatrix);
            }

            //Start Given's rotation
            //Example from: https://en.wikipedia.org/wiki/Givens_rotation                        
            //Apply Given's rotation from left

            //Parallel.For(0, hMatrix.Length,
            //   new ParallelOptions { MaxDegreeOfParallelism = MaxParallelThreads },
            //   col =>
            if(U == null)
                U = IdentityMatrix(hMatrix.Length);

            double[][][] G = new double[hMatrix.Length][][];
            for (int col=0;col<hMatrix.Length-1;col++)
            {                   
                   G[col] = IdentityMatrix(hMatrix.Length);
                    //Compute r
                    double tempR = System.Math.Pow(hMatrix[col][col], 2) +
                                System.Math.Pow(hMatrix[col][col + 1], 2);
                    tempR = System.Math.Sqrt(tempR);
                    double tempC = hMatrix[col][col] / tempR;
                    double tempS = -hMatrix[col][col + 1] / tempR;
                    G[col][col][col] = tempC;
                    G[col][col + 1][col + 1] = tempC;
                    G[col][col + 1][col] = -tempS;
                    G[col][col][col + 1] = tempS;
                    hMatrix = Multiply(G[col], hMatrix);
                    //U = Multiply(U, Transpose(G[col]));

                if (col == 0)
                    tmpQ = Transpose(G[col]);
                else
                    tmpQ = Multiply(tmpQ, Transpose(G[col]));



            }//);
            //Apply Given's rotation from right

            //for (int col = 0; col < hMatrix.Length - 1; col++)
            //{                            
            //    hMatrix = Multiply(hMatrix, G[col]);                
            //}        
            Q = tmpQ;
            R = hMatrix;  
            
        } //QRDecomposition_Givens
    }
}
