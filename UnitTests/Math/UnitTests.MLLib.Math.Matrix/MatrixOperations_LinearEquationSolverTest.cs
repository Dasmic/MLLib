using Microsoft.VisualStudio.TestTools.UnitTesting;
using Dasmic.MLLib.Math.Matrix;
using Dasmic.MLLib.Common.Exceptions;
using Dasmic.Portable.Core;

namespace UnitTests.MLLib.Algorithms.Matrix
{
    public partial class MatrixOperationsTest
    {

        /// <summary>
        /// Example taken from:
        /// https://programmingpraxis.com/2010/07/20/solving-systems-of-linear-equations/
        /// </summary>
        [TestMethod]
        public void Matrix_solver_linear_3_row_test()
        {
            double[][] matrixA = new double[3][];
            matrixA[0] = new double[] { 1,3,5};
            matrixA[1] = new double[] { 2, 5, 6 };
            matrixA[2] = new double[] { 0, 4, 3 };

            double[][] matrixB = new double[1][];
            matrixB[0] = new double[] { 1.0/10.0,25.0/2.0,103.0/10.0 };
            
            

            MatrixOperations mo = new MatrixOperations();
            double[][] matrixX=mo.SolveLinearEquation(matrixA, matrixB);

            Assert.IsTrue(SupportFunctions.DoubleCompare(matrixX[0][0], 1.0/2.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(matrixX[0][1], -1.0 / 5.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(matrixX[0][2], 3.0));
        }


        [TestMethod]
        public void Matrix_solver_linear_2_row_test()
        {
            double[][] matrixA = new double[2][];
            matrixA[0] = new double[] { 2, 5 };
            matrixA[1] = new double[] { 1, -2 };
            

            double[][] matrixB = new double[1][];
            matrixB[0] = new double[] { 4,1 };

            
            MatrixOperations mo = new MatrixOperations();
            double[][] matrixX=mo.SolveLinearEquation(matrixA, matrixB);

            Assert.IsTrue(SupportFunctions.DoubleCompare(matrixX[0][0], 1.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(matrixX[0][1], 2.0));
            
        }
    }
}

