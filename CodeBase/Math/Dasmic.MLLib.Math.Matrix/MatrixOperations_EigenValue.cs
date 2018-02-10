using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Dasmic.MLLib.Math.Matrix
{
    public partial class MatrixOperations
    {
        // General Notes:
        //A.I = A; I.A = A
        //A.v = E.v => A.v - I.E.v = 0 
        // => (A-I.E).v = 0. Since v is non 0, (A-I.E) is not-invertible
        // => det(A-I.E) = 0 or (A-I.E) is singular

        /// <summary>
        /// Finds EigenValues and EigenVectors using QR Algorithm
        /// for a Square Matrix.
        /// 
        /// p()=det(A−I.E), where E is Eignevalue of matrixA
        /// p() is characteristic polynomial.
        /// 
        /// This algorithm will find value of E, where p()=0
        /// or (A−I.E) is a singular matrix
        /// </summary>
        /// <param name="matrix">Matrix whose EigenValue is to be found</param>
        /// <param name="eigenValues">Vector to store computed EigenValues </param>
        /// <param name="eigenVectors">Vector to store computed EigenVectors</param>
        /// <param name="qrMethod">Method to use for QR Decomposition. "Householder" or "Givens" </param>
        /// <param name="maxIterations">Maximum iterations for QR Algorithm</param>
        /// <returns>None. All values are passed via array references</returns>
        public void EigenValues(double[][] matrix,
                                ref double[] eigenValues,
                                ref double[][] eigenVectors,
                                string qrMethod="Householder",
                                int maxIterations = 1000)
        {
            GeneralVerifyAndIfSquare(matrix);
           
            //Do QR Decomposition
            double[][] Q = null;
            double[][] R = null;

            bool isGivens = qrMethod.Trim().ToUpper().Equals("GIVENS") ? true : false;
            //Create copy
            double[][] M=null;
            CopyMatrix(matrix, ref M);
            //double[][] hessenbergU = null;
            //M = Hessenberg(matrix, ref hessenbergU);
                    
            int iter = 0;
            double[][] eigenMatrix = IdentityMatrix(M.Length);
            //double[][] qrDecompU = IdentityMatrix(M.Length);

            //This is the QR Algorithm
            //Now start convergence loop
            //While matrix is not a Diagonal Matrix 
            while (!IsDiagonal(M) && iter++<maxIterations)
            {
                if(isGivens)
                    QRDecomposition_Givens_Rotation(M, ref Q, ref R, false);
                else
                    QRDecomposition_Householder(M, ref Q, ref R, false);                
                
                //double[][] newQ = Multiply(U, Q);
                //newQ = Multiply(newQ, Transpose(U));
                //double[][] newR = Multiply(R, U);
                //double[][] newM = Multiply(Q, R);
                //if (!CompareMatrix(M, newM)) 
                //    throw new Exception();
                M = Multiply(R, Q);
                //if(U != null)
                //    qrDecompU = Multiply(qrDecompU, U);
                eigenMatrix = Multiply(eigenMatrix,Q);//eigenMatrix,Q);                
            }

            //if (!IsDiagonal(M))
            //    throw new Exception();
            //Start Loop to extract EigenValues
            //Columns of Q are EigenVectors
            //Diagonal elements of R are EigenValues
            double[] eigValues = new double[R.Length];
            Parallel.For(0, M.Length,
               new ParallelOptions { MaxDegreeOfParallelism = MaxParallelThreads },
               col =>
               //for (int col=0;col<R.Length;col++)
               {
                eigValues[col] = M[col][col];
               });

            //newQ = Multiply(U, newQ);
            eigenValues = eigValues;            
            eigenVectors =  eigenMatrix;            
        }
    }
}
