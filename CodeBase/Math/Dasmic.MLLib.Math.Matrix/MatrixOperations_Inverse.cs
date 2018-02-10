using Dasmic.MLLib.Common.Exceptions;

namespace Dasmic.MLLib.Math.Matrix
{
    public partial class MatrixOperations
    {
        //This is for a general matrix
        //Row/Col concepts can differ from those used in trainingData


        /// <summary>
        /// Compute inverse of the Matrix
        /// </summary>
        /// <param name="matrix"></param>
        /// <returns></returns>
        public double[][] Inverse(double[][]
                                matrix)
        {
            VerifyMatrix(matrix);

            //Does not check if determinant is 0, 
            //that check is external to this       
            //Check if square matrix
            if (matrix[0].Length != matrix.Length)
                throw new InvalidMatrixException();
            
            double[][] matrixI = IdentityMatrix(matrix.Length);
            double[][] newMatrix = SolveLinearEquation(matrix, matrixI);
            
            return newMatrix;
        }
    }
}
