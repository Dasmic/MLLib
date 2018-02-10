using Microsoft.VisualStudio.TestTools.UnitTesting;
using Dasmic.MLLib.Math.Matrix;
using Dasmic.Portable.Core;



namespace UnitTests.MLLib.Algorithms.Matrix
{
    public partial class MatrixOperationsTest
    {
        /// <summary>
        /// Results from:
        /// https://matrix.reshish.com/inverse.php
        /// </summary>
        [TestMethod]
        public void Matrix_inverse_2_row_test()
        {

            double[][] matrixA = new double[2][];
            matrixA[0] = new double[] { 1, 2 };
            matrixA[1] = new double[] { 3, 4 };

            MatrixOperations mo = new MatrixOperations();
            double[][] matrixI =
                            mo.Inverse(matrixA);

            //Check Det
            Assert.IsTrue(SupportFunctions.DoubleCompare(matrixI[0][0], -2.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(matrixI[0][1], 1.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(matrixI[1][0], 3.0/2.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(matrixI[1][1], -1.0/2.0));           
        }

        /// <summary>
        /// Results from:
        /// 
        /// https://matrix.reshish.com/inverse.php
        /// </summary>
        [TestMethod]
        public void Matrix_inverse_determinant_0_3_row_test()
        {

            double[][] matrixA = new double[3][];
            matrixA[0] = new double[] { 1, 4, 7 };
            matrixA[1] = new double[] { 2, 5, 8 };
            matrixA[2] = new double[] { 3, 6, 9 };

            MatrixOperations mo = new MatrixOperations();
            double[][] matrixI =
                            mo.Inverse(matrixA);

            //Check Det
            Assert.IsFalse(SupportFunctions.DoubleCompare(matrixI[0][0], -11.0/12.0));
                       
        }


        /// <summary>
        /// Results from:
        /// 
        /// http://onlinemschool.com/math/assistance/matrix/inverse/
        /// </summary>
        [TestMethod]
        public void Matrix_inverse_3_row_test()
        {

            double[][] matrixA = new double[3][];
            matrixA[0] = new double[] { 1, 2, 3 };
            matrixA[1] = new double[] { 4, 0, 6 };
            matrixA[2] = new double[] { 7, 8, 9 };

            MatrixOperations mo = new MatrixOperations();
            double[][] matrixI =
                            mo.Inverse(matrixA);

            //Check Det
            Assert.IsTrue(SupportFunctions.DoubleCompare(matrixI[0][0], -0.8));
            Assert.IsTrue(SupportFunctions.DoubleCompare(matrixI[0][1], 0.1));
            Assert.IsTrue(SupportFunctions.DoubleCompare(matrixI[0][2], 0.2));
            Assert.IsTrue(SupportFunctions.DoubleCompare(matrixI[1][0], 0.1));
            Assert.IsTrue(SupportFunctions.DoubleCompare(matrixI[1][1], -0.2));
            Assert.IsTrue(SupportFunctions.DoubleCompare(matrixI[1][2], 0.1));
            Assert.IsTrue(SupportFunctions.DoubleCompare(matrixI[2][0], 8.0/15.0));
            Assert.IsTrue(SupportFunctions.DoubleCompare(matrixI[2][1], 0.1));
            Assert.IsTrue(SupportFunctions.DoubleCompare(matrixI[2][2], -2.0/15.0));

        }

    }
}
