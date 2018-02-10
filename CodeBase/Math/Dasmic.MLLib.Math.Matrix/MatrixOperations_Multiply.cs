using System.Threading.Tasks;
using Dasmic.MLLib.Common.Exceptions;

namespace Dasmic.MLLib.Math.Matrix
{
    public partial class MatrixOperations
    {
        /// <summary>
        /// Multiple 2 matrices and return result matrix
        /// 
        /// MatrixA is on left and MatrixB is on right
        /// </summary>
        /// <param name="matrixA"></param>
        /// <param name="matrixB"></param>
        /// <returns></returns>
        public double[][] Multiply(double[][]
                                matrixA, 
                                double[][] matrixB)
        {
            VerifyMatrix(matrixA);
            VerifyMatrix(matrixB);

            //Check multiply semantics
            //Check cols in 1 is same as rows in other
            //Multiplying row1*col1 into row2*col2 leads to row1*col2 matrix
            if (matrixA.Length != matrixB[0].Length)
                throw new MatrixInputMismatchException();

            double[][] newMatrix =
                new double[matrixB.Length][];

            //Add new arrays
            //for (int idx = 0; idx < newMatrix.Length; idx++)
            Parallel.For(0, newMatrix.Length,
                new ParallelOptions { MaxDegreeOfParallelism = MaxParallelThreads },
                idx =>
            {
                newMatrix[idx] = new double[matrixA[0].Length];
            });

            //Start Multiply Op
            //Row here is used for clarity to denote row size of tMatrix, 
            //it may differ from actual row
            //same applies to col
            //double sum;

            //for (col = 0; col < newMatrix.Length; col++)
            Parallel.For(0, newMatrix.Length,
                new ParallelOptions { MaxDegreeOfParallelism = MaxParallelThreads },
                col =>
            //for (int col = 0; col < newMatrix.Length; col++)
            {
                int row = 0;
                for (row = 0; row < newMatrix[0].Length; row++)
                {
                    double sum = 0; //Each thread will have own copy
                    for (int ii = 0; ii < matrixA.Length; ii++)
                        sum += matrixA[ii][row] * matrixB[col][ii];

                    newMatrix[col][row] = sum;
                }
            });
            return newMatrix;
        }
    }
}
