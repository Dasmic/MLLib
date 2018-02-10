using System;
using System.Threading.Tasks;
using Dasmic.MLLib.Common.Exceptions;

namespace Dasmic.MLLib.Math.Matrix
{
    public partial class MatrixOperations
    {
        /// <summary>
        /// Solve for system of linear equations in matrix form:
        /// MatrixA * MatrixX = MatrixB
        /// 
        /// Will retunn the solution matrix, MatrixX
        /// </summary>
        /// <param name="matrixA"></param>
        /// <param name="matrixB"></param>
        /// <param name="matrixX"></param>
        public double[][] SolveLinearEquation(double[][] matrixA,
                                    double[][] matrixB)
        {
            //Do Validation
            if (matrixB.Length < 1 )
                throw new InvalidMatrixException();
            
            //Same columns in X and B Matrix
            if (matrixB[0].Length != matrixA.Length)
                throw new MatrixInputMismatchException ();

            //Cols != Rows
            if (matrixB[0].Length != matrixA[0].Length)
                throw new MatrixInputMismatchException();

            double[][] matrixX = new double[matrixB.Length][];
            //for (int idx = 0; idx < matrixB.Length; idx++)
            Parallel.For (0, matrixB.Length, idx=>
            {
                matrixX[idx] = new double[matrixB[0].Length];
            });

            //A.X = B
            //Get LUP factorization
            double[][] L = null, U = null, P = null;
            LupMatrix(matrixA, ref L, ref U, ref P);
            
            //P.A = L.U => A = L.U.P-1
            //L.U.P-1.X = B => L.U.X = B/P-1
            //=> L.U.X = P.B
            double[][] pMatrixB = Multiply(P, matrixB);

            //For each column
            //for (int colIdx = 0; colIdx < matrixX.Length; colIdx++)
            Parallel.For(0, matrixX.Length, colIdx =>
             {
                //Do forward substitution
                //L.(U.X) = P.B,  Y = U.X
                //L.Y = P.B
                //These steps have to be sequential so do not parallelize
                double[] Y = new double[matrixA[0].Length];
                 for (int rowIdx = 0; rowIdx < Y.Length; rowIdx++)
                 {
                    //Get Sum
                    double sum = 0;
                     for (int idx = 0; idx < rowIdx; idx++)
                     {
                         sum += L[idx][rowIdx] * Y[idx];
                     }

                     Y[rowIdx] = (1 / L[rowIdx][rowIdx]) *
                                     (pMatrixB[colIdx][rowIdx] - sum);
                 }

                //Do backward substitution
                //L.Y = P.B,  Y = U.X
                //we know Y, U.X = Y
                for (int rowIdx = Y.Length - 1; rowIdx >= 0; rowIdx--)//(int rowIdx = 0; rowIdx < Y.Length; rowIdx++)
                {
                    //Get Sum
                    double sum = 0;
                     for (int idx = rowIdx + 1; idx < Y.Length; idx++)
                     {
                         sum += U[idx][rowIdx] * matrixX[colIdx][idx];
                     }

                     matrixX[colIdx][rowIdx] = (1 / U[rowIdx][rowIdx]) *
                                     (Y[rowIdx] - sum);
                 }//Next colIdx


            });
            return matrixX;
        }
    }
}
