using Microsoft.VisualStudio.TestTools.UnitTesting;
using Dasmic.MLLib.Math.Matrix;
using Dasmic.MLLib.Common.Exceptions;

namespace UnitTests.MLLib.Algorithms.Matrix
{
    public partial class MatrixOperationsTest
    {
        #region Multiply
        [TestMethod]
        public void Matrix_multiply_2_row_test()
        {
            InitData_dataset_2_rows();
            double[][] newMatrix =
                _mo.Multiply(_matrix1, _matrix2);

            Assert.AreEqual(newMatrix.Length, 2);
            Assert.AreEqual(newMatrix[0].Length, 2);

            Assert.AreEqual(newMatrix[0][0], 23);
            Assert.AreEqual(newMatrix[0][1], 34);
            Assert.AreEqual(newMatrix[1][0], 31);
            Assert.AreEqual(newMatrix[1][1], 46);
        }

        [TestMethod]
        public void Matrix_multiply_4_row_test()
        {
            InitData_dataset_4_rows_symmetric_hessenberg();
            double[][] newMatrix =
                _mo.Multiply(_matrix1, _matrix2);

            Assert.AreEqual(newMatrix.Length, 4);
            Assert.AreEqual(newMatrix[0].Length, 4);

            Assert.AreEqual(newMatrix[0][0], 25);
            Assert.AreEqual(newMatrix[0][1], 8);
            Assert.AreEqual(newMatrix[0][2], -18);
            Assert.AreEqual(newMatrix[0][3], 11);

            Assert.AreEqual(newMatrix[1][0], 8);
            Assert.AreEqual(newMatrix[1][1], 6);
            Assert.AreEqual(newMatrix[1][2], -4);
            Assert.AreEqual(newMatrix[1][3], 3);

            Assert.AreEqual(newMatrix[2][0], -18);
            Assert.AreEqual(newMatrix[2][1], -4);
            Assert.AreEqual(newMatrix[2][2], 17);
            Assert.AreEqual(newMatrix[2][3], -8);

            Assert.AreEqual(newMatrix[3][0], 11);
            Assert.AreEqual(newMatrix[3][1], 3);
            Assert.AreEqual(newMatrix[3][2], -8);
            Assert.AreEqual(newMatrix[3][3], 10);
        }


        /// <summary>
        /// Results:
        /// http://www.calcul.com/show/calculator/matrix-multiplication
        /// </summary>
        [TestMethod]
        public void Matrix_multiply_nonsquare_row_test()
        {
            InitData_dataset_nonsquare_matrix();
            double[][] newMatrix =
                _mo.Multiply(_matrix1, _matrix2);

            Assert.AreEqual(newMatrix.Length, 3);
            Assert.AreEqual(newMatrix[0].Length, 3);

            Assert.AreEqual(newMatrix[0][0], 23);
            Assert.AreEqual(newMatrix[0][1], 53);
            Assert.AreEqual(newMatrix[0][2], 83);

            Assert.AreEqual(newMatrix[1][0], 29);
            Assert.AreEqual(newMatrix[1][1], 67);
            Assert.AreEqual(newMatrix[1][2], 105);

            Assert.AreEqual(newMatrix[2][0], 35);
            Assert.AreEqual(newMatrix[2][1], 81);
            Assert.AreEqual(newMatrix[2][2], 127);
        }


        [TestMethod]
        public void Matrix_multiply_nonsquare_single_column_test()
        {

            double[][] _matrix1 = new double[3][];
            _matrix1[0] = new double[] { 0, 1, 0 };
            _matrix1[1] = new double[] { 0, 0, 1 };
            _matrix1[2] = new double[] { 1, 0, 0 };

            double[][] _matrix2 = new double[1][];
            _matrix2[0] = new double[] { 1, 2, 3 };
            

            double[][] newMatrix =
                _mo.Multiply(_matrix1, _matrix2);

            Assert.AreEqual(newMatrix.Length, 1);
            Assert.AreEqual(newMatrix[0].Length, 3);

            Assert.AreEqual(newMatrix[0][0], 3);
            Assert.AreEqual(newMatrix[0][1], 1);
            Assert.AreEqual(newMatrix[0][2], 2);

        }

        #endregion
    }
}
