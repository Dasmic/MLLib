using System.Threading.Tasks;
using Dasmic.MLLib.Common.Exceptions;

namespace Dasmic.MLLib.Math.Matrix
{
    public partial class MatrixOperations
    {
        /// <summary>
        /// Substract 2 matrices and return result matrix
        /// 
        /// Both matrices should have same dimension
        /// MatrixA is on left and MatrixB is on right
        /// </summary>
        /// <param name="matrixA"></param>
        /// <param name="matrixB"></param>
        /// <returns></returns>
        public double[][] Substract(double[][]
                                matrixA, 
                                double[][] matrixB)
        {
            //No verification of matrices needed

            //Check matric correctness
            
            if (matrixA.Length != matrixB.Length)
                throw new MatrixInputMismatchException();
            if (matrixA[0].Length != matrixB[0].Length)
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

            //Start  Op           
            //for (col = 0; col < newMatrix.Length; col++)
            Parallel.For(0, newMatrix.Length,
                new ParallelOptions { MaxDegreeOfParallelism = MaxParallelThreads },
                col =>
            //for (int col = 0; col < newMatrix.Length; col++)
            {
                int row = 0;
                for (row = 0; row < newMatrix[0].Length; row++)
                {
                    newMatrix[col][row] = matrixA[col][row]
                                            - matrixB[col][row];
                }
            });
            return newMatrix;
        }
    }
}
