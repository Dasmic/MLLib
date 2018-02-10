using Microsoft.VisualStudio.TestTools.UnitTesting;
using Dasmic.MLLib.Math.Matrix;
using Dasmic.MLLib.Common.Exceptions;

namespace UnitTests.MLLib.Algorithms.Matrix
{
    public partial class MatrixOperationsTest
    {
        #region Transpose
        [TestMethod]
        public void Matrix_transpose_2_row_test()
        {
            InitData_dataset_2_rows();
            double[][] newMatrix =
                _mo.Transpose(_matrix1);

            Assert.AreEqual(newMatrix[0][0], 1);
            Assert.AreEqual(newMatrix[0][1], 3);
            Assert.AreEqual(newMatrix[1][0], 2);
            Assert.AreEqual(newMatrix[1][1], 4);
        }

        [TestMethod]
        public void Matrix_transpose_3_row_test()
        {
            InitData_dataset_3_rows_non_symmetric();
            double[][] newMatrix =
                _mo.Transpose(_matrix1);

            Assert.AreEqual(newMatrix[0][0], 1);
            Assert.AreEqual(newMatrix[0][1], 4);
            Assert.AreEqual(newMatrix[0][2], 7);
            Assert.AreEqual(newMatrix[1][0], 2);
            Assert.AreEqual(newMatrix[1][1], 5);
            Assert.AreEqual(newMatrix[1][2], 8);
            Assert.AreEqual(newMatrix[2][0], 3);
            Assert.AreEqual(newMatrix[2][1], 6);
            Assert.AreEqual(newMatrix[2][2], 9);
        }

        [TestMethod]
        public void Matrix_transpose_3_row_non_square_test()
        {
            InitData_dataset_3_rows_non_square();
            double[][] newMatrix = 
                _mo.Transpose(_matrix1);

            Assert.AreEqual(newMatrix.Length, 2);
            Assert.AreEqual(newMatrix[0].Length, 3);
            Assert.AreEqual(newMatrix[0][0], 1);
            Assert.AreEqual(newMatrix[0][1], 3);
            Assert.AreEqual(newMatrix[0][2], 5);
            Assert.AreEqual(newMatrix[1][0], 2);
            Assert.AreEqual(newMatrix[1][1], 4);
            Assert.AreEqual(newMatrix[1][2], 6);
            
        }


        #endregion


        [TestMethod]
        [ExpectedException(typeof(InvalidMatrixException))]
        public void Matrix_transpose_invalid_data_throws_exception()
        {
            InitData_dataset_invalid();
            double[][] newMatrix =
                _mo.Transpose(_matrix1);
        }

    }
}
